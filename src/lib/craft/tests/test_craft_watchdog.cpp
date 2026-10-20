/*********************************************************************************
 * Modifications Copyright 2026 eBay Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 *********************************************************************************/

// Unit tests for CraftReplDev's client-liveness watchdog (S7).
//
// Scheduling a real iomgr timer needs a running reactor pool -- the only reason this is its own
// binary rather than living in test_craft_commit.cpp: every other light CRAFT test compiles
// craft_repl_dev.cpp directly with a mock journal and NEVER starts iomgr (they set
// craft_watchdog_timeout_ms=0 so none of them need to). This file starts a minimal, HomeStore-free iomgr
// instance (see main()) so touch_watchdog()/on_watchdog_tick() can run for real, without pulling
// in the much heavier HomeStore bring-up test_craft_homestore_backend.cpp needs for its own real
// home_log_store -- and, unlike that file, this one compiles craft_repl_dev.cpp directly with
// _PRERELEASE (matching test_craft_write.cpp/test_craft_commit.cpp/test_craft_truncate.cpp/
// test_craft_peer_exchange.cpp), so seed_term()/watchdog_fire_count() are actually available.
//
// This TU defines SISL_LOGGING_DEF for the homeblocks module because it compiles craft_repl_dev.cpp
// directly (same pattern as test_craft_truncate.cpp).

#include <atomic>
#include <chrono>
#include <thread>

#include <gtest/gtest.h>
#include <sisl/logging/logging.h>
#include <sisl/options/options.h>
#include <iomgr/iomgr.hpp>

#include "craft/craft_repl_dev.hpp"
#include "home_blks_config.hpp"
#include "coro_helpers.hpp"
#include "mock_journal_backend.hpp"

SISL_LOGGING_DEF(HOMEBLOCKS_LOG_MODS)
SISL_OPTIONS_ENABLE(logging)
SISL_LOGGING_INIT(HOMEBLOCKS_LOG_MODS)

namespace homeblocks {
namespace {

static constexpr uint32_t k_page_size = 4096;
static constexpr uint64_t k_test_watchdog_timeout_ms = 300; // ms; overrides craft_watchdog_timeout_ms in SetUp

// ── minimal journal mock ────────────────────────────────────────────────────────
//
// The watchdog never touches journal_ for data: the sync it triggers only Empty-verdicts slots nobody
// holds, so every method is an unreachable stub, same shape as test_craft_truncate.cpp's mock.

class MockCraftJournalBackend : public CraftJournalBackend {
public:
    async_result< homestore::multi_blk_id > alloc_write_data(sisl::sg_list const&, lba_count_t) override {
        co_return std::unexpected(std::make_error_condition(std::errc::not_supported));
    }
    async_status write_slot(int64_t, uint64_t, lba_t, lba_count_t, homestore::multi_blk_id, bool,
                            std::vector< homestore::csum_t > const&) override {
        co_return std::unexpected(std::make_error_condition(std::errc::not_supported));
    }
    async_result< JournalSlot > read_slot(int64_t) override {
        co_return std::unexpected(std::make_error_condition(std::errc::not_supported));
    }
    async_status truncate_to(int64_t) override { co_return ok(); }
    async_status free_data(homestore::multi_blk_id) override { co_return ok(); }
    async_status free_slot(int64_t) override {
        co_return std::unexpected(std::make_error_condition(std::errc::not_supported));
    }
    async_status read_data(homestore::multi_blk_id, sisl::sg_list&) override {
        co_return std::unexpected(std::make_error_condition(std::errc::not_supported));
    }
};

// ── peer fetcher mock ─────────────────────────────────────────────────────────
//
// poll_quorum_lsns returns `polled` (one entry per responding peer); fetch_from_quorum reports one
// responder holding nothing, so any slot beyond the leader's frontier is verdicted Empty. Called on the
// reactor thread: tests read state only after dev_.reset() (cancel(wait=true) joins any in-flight tick).

class MockCraftQuorumPoller : public CraftPeerFetcher {
public:
    std::vector< craft::lsn_pair > polled;
    std::atomic< int > poll_calls{0};

    async_result< craft::lsn_pair > get_rs_commit_lsn(uint64_t, bool) override { co_return craft::lsn_pair{}; }
    async_result< std::vector< JournalSlot > > fetch_data(const std::vector< int64_t >&, uint32_t) override {
        co_return std::unexpected(std::make_error_condition(std::errc::not_supported));
    }
    async_result< std::vector< QuorumSlotResponse > > fetch_from_quorum(std::vector< int64_t >, uint32_t) override {
        co_return std::vector< QuorumSlotResponse >{QuorumSlotResponse{}};
    }
    async_result< std::vector< craft::lsn_pair > > poll_quorum_lsns(uint64_t, bool, uint32_t) override {
        ++poll_calls;
        co_return polled;
    }
};

// ── test fixture ─────────────────────────────────────────────────────────────

class CraftWatchdogTest : public ::testing::Test {
protected:
    void SetUp() override {
        orig_timeout_ms_ = HB_DYNAMIC_CONFIG(craft_watchdog_timeout_ms);
        HB_SETTINGS_FACTORY().modifiable_settings(
            [](auto& s) { s.craft_watchdog_timeout_ms = k_test_watchdog_timeout_ms; });
        dev_ = CraftReplDev::create(volume_id_t{}, std::make_unique< MockCraftJournalBackend >(), k_page_size,
                                    /* indx_tbl = */ nullptr);
    }

    void TearDown() override {
        dev_.reset();
        HB_SETTINGS_FACTORY().modifiable_settings([this](auto& s) { s.craft_watchdog_timeout_ms = orig_timeout_ms_; });
    }

    // keep_alive() with a matching term is the only public path that both succeeds (term check
    // passes) and calls touch_watchdog() -- write() would work too, but needs a real data payload
    // for the non-all_zeros case; keep_alive() needs nothing but the header.
    auto do_keep_alive(uint64_t term) {
        return homeblocks::detail::sync_get(dev_->keep_alive(craft::client_hdr{term, -1, -1}));
    }

    // Wires a leader proposer + quorum poller and logs in at `term` (keep_alive arms the watchdog).
    void arm_as_leader(uint64_t term) {
        dev_->set_peer_fetcher(&fetcher_);
        dev_->set_raft_proposer(&proposer_);
        dev_->seed_term(term);
        ASSERT_TRUE(do_keep_alive(term).has_value());
    }

    uint64_t orig_timeout_ms_{0};
    MockCraftQuorumPoller fetcher_;
    MockCraftRaftProposer proposer_;
    std::shared_ptr< CraftReplDev > dev_;
};

// ── tests ─────────────────────────────────────────────────────────────────────

// No proposer is wired here, so watchdog_fire_count() (test-only) is how this confirms
// on_watchdog_tick() actually fired after a full interval of no write()/keep_alive() activity.
TEST_F(CraftWatchdogTest, FiresAfterInactivity) {
    dev_->seed_term(7);

    auto r = do_keep_alive(7);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(dev_->watchdog_fire_count(), 0); // not yet -- keep_alive() itself just armed the timer

    std::this_thread::sleep_for(std::chrono::milliseconds(600)); // > 300ms configured timeout
    // A small extra wait immediately before the assertion: 600ms is comfortably past when the timer
    // should have ticked, but doesn't guarantee the iomgr worker thread has actually finished running
    // that tick's callback by the moment THIS thread wakes from its own sleep -- those are two
    // independently scheduled threads. Without this, the check is a real (if narrow) flakiness risk on
    // a loaded machine.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_GE(dev_->watchdog_fire_count(), 1);
}

// A keep_alive() before the interval elapses must refresh last_contact_ns_ -- no firing as long as
// activity keeps arriving faster than the timeout (each recurring tick sees a recent timestamp and
// takes no action).
TEST_F(CraftWatchdogTest, DoesNotFireIfResetInTime) {
    dev_->seed_term(7);

    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(do_keep_alive(7).has_value());
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // well under the 300ms timeout
    }
    EXPECT_EQ(dev_->watchdog_fire_count(), 0);
}

// A not-yet-logged-in partition (term still 0, the default) must not arm the watchdog at all --
// there is no session to watch yet.
TEST_F(CraftWatchdogTest, NoLoginNeverArms) {
    // hdr.term=0 matches state_.term's default (0), so keep_alive() itself succeeds -- this
    // specifically tests touch_watchdog()'s own state_.term==0 guard, not the term-check rejection.
    ASSERT_TRUE(do_keep_alive(0).has_value());

    std::this_thread::sleep_for(std::chrono::milliseconds(600)); // > 300ms configured timeout
    EXPECT_EQ(dev_->watchdog_fire_count(), 0);
}

// ── leader sync (SDSTOR-22908) ────────────────────────────────────────────────

// A stale tick on the leader polls the quorum and syncs to its max last_append -- once: later stale
// ticks find that watermark already proposed.
TEST_F(CraftWatchdogTest, LeaderStaleTickProposesQuorumMaxOnce) {
    dev_->seed_lsns(5, {});
    fetcher_.polled = {craft::lsn_pair{.commit_lsn = -1, .last_append_lsn = 8}};
    arm_as_leader(7);

    std::this_thread::sleep_for(std::chrono::milliseconds(1200)); // several stale ticks past the 300ms timeout
    dev_.reset(); // joins any in-flight tick before the mocks are read

    EXPECT_GE(fetcher_.poll_calls.load(), 2);
    ASSERT_EQ(proposer_.proposed.size(), 1u);
    auto const e = decode(proposer_.proposed[0]);
    EXPECT_EQ(e.rs_commit_lsn, 8);
    EXPECT_EQ(e.empty_slots, (std::vector< int64_t >{6, 7, 8})); // beyond every frontier, held by nobody
}

TEST_F(CraftWatchdogTest, FollowerStaleTickNeverPolls) {
    proposer_.leader = false;
    arm_as_leader(7);

    std::this_thread::sleep_for(std::chrono::milliseconds(620));
    EXPECT_GE(dev_->watchdog_fire_count(), 1);
    dev_.reset();

    EXPECT_EQ(fetcher_.poll_calls.load(), 0);
    EXPECT_TRUE(proposer_.proposed.empty());
}

// N=5 needs self + 2 responders; one isn't enough to sync.
TEST_F(CraftWatchdogTest, StaleTickWithoutQuorumDoesNotPropose) {
    proposer_.members = 5;
    fetcher_.polled = {craft::lsn_pair{.commit_lsn = -1, .last_append_lsn = 8}};
    arm_as_leader(7);

    std::this_thread::sleep_for(std::chrono::milliseconds(620));
    dev_.reset();

    EXPECT_GE(fetcher_.poll_calls.load(), 1);
    EXPECT_TRUE(proposer_.proposed.empty());
}

// Destroys a CraftReplDev at an unpredictable point relative to its own watchdog timer's firing --
// sometimes well before the first tick, sometimes squarely inside the window where on_watchdog_tick()
// is either about to run, actively running, or has just returned. An EARLIER, one-shot-timer-that-
// reschedules-itself design produced a real, reproduced SEGFAULT here (heap corruption in iomgr's
// timer heap from the self-reschedule cancelling its own already-fired, already-consumed handle) --
// the current design uses a genuinely RECURRING iomgr::timer_token instead specifically to remove
// that whole hazard class (see touch_watchdog()/on_watchdog_tick()'s doc comments), but this test
// stays as a regression guard against any future variant of the same race, under either design. Many
// short-lived instances with a very short timeout, each racing destruction against a real timer
// firing, maximize the chance of hitting the narrow window; a single iteration would very likely pass
// even with a reintroduced bug.
TEST_F(CraftWatchdogTest, DestructorRacesFiringManyIterations) {
    // Override to 2ms -- deliberately far shorter than SetUp()'s 300ms so firing is near-certain
    // before destruction. Restores SetUp()'s value on scope exit via TearDown (fixture resets it).
    HB_SETTINGS_FACTORY().modifiable_settings([](auto& s) { s.craft_watchdog_timeout_ms = 2; });
    static constexpr int kIterations = 200;

    for (int i = 0; i < kIterations; ++i) {
        auto dev = CraftReplDev::create(volume_id_t{}, std::make_unique< MockCraftJournalBackend >(), k_page_size,
                                        /* indx_tbl = */ nullptr);
        dev->seed_term(7);
        ASSERT_TRUE(homeblocks::detail::sync_get(dev->keep_alive(craft::client_hdr{7, -1, -1})).has_value())
            << "iteration " << i;
        // No sleep here: destruction races the timer on purpose, rather than waiting it out first --
        // that is the entire point of this test (see the doc comment above).
        dev.reset();
    }
    // Reaching here at all (no crash, no hang) across every iteration is this test's actual
    // assertion -- there is no separate observable count to check beyond survival.
    SUCCEED();
}

} // namespace
} // namespace homeblocks

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // iomanager.start() constructs iomgr's own settings factory, which needs SISL_OPTIONS parsed
    // first (crashes otherwise) -- every other light CRAFT test skips this because none of them
    // call iomanager.start() at all.
    SISL_OPTIONS_LOAD(argc, argv, logging);
    sisl::logging::SetLogger("test_craft_watchdog");
    spdlog::set_pattern("[%D %T%z] [%^%l%$] [%n] [%t] %v");

    // Initialize the HomeBlks settings factory with flatbuffers defaults so HB_DYNAMIC_CONFIG()
    // and HB_SETTINGS_FACTORY().modifiable_settings() work without a live HomeBlks instance.
    HB_SETTINGS_FACTORY().load_json("{}");

    // No HomeStore: just enough of a real iomgr reactor pool for iomgr::schedule_recurring/
    // timer_token::cancel to work. Matches iomgr's own minimal test bring-up (src/test/test_timer.cpp).
    iomanager.start(iomgr::iomgr_params{.num_threads = 2});
    auto ret = RUN_ALL_TESTS();
    iomanager.stop();
    return ret;
}
