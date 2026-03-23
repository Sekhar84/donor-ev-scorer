"""
BelFund Donor EV Scorer — Data Simulation
==========================================
Generates synthetic donor data calibrated to real Belgian nonprofit
direct mail fundraising distributions. Used for development and testing
when real donor data is unavailable or cannot be shared.

Campaign type: FID (Fidelization / loyalty campaigns)
  — Targets active donors who have given at least once in the past 24 months.
  — Goal: sustain and grow giving from already-engaged donors.
  — Selection logic: mail donor if EV > cost_per_mail.

All client identity is loaded from environment variables (.env).
No real client identifiers appear in this script.

Calibration source: profiling of a real FID campaign dataset.

Key distributions matched:
  - 6.53% overall response rate
  - Amount: median €35, mean €39, log-normal (mu=3.24, sigma=0.93)
  - Amount cap at p90=€65, p95=€100
  - 19 FID campaigns, campaign-level RR range 2-10%
  - Gifts per donor: median=2, mean=7.86, highly right-skewed
  - Days since last gift: median=1,420 — 33% FID eligible (<730d)
  - SDD: 1.39% of selection pool, Monthly type dominant (83%)
  - OP standing orders: 0.32% of selection pool
  - Language: Dutch 66%, French 34%
  - Gender: Female 46%, Male 27%, Couple 23%, Unknown 4%
  - Cost per mail: median €1.10, mean €1.09

Run:
  python src/simulate.py

Outputs written to ./data/simulated/
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config from environment — no hardcoded client values ──────────────────
CLIENT_ID    = int(os.getenv("CLIENT_ID", "47"))
CLIENT_NAME  = os.getenv("CLIENT_NAME", "BelFund")
SIM_SEED     = int(os.getenv("SIM_SEED", "42"))
N_DONORS     = int(os.getenv("SIM_N_DONORS", "15000"))
N_CAMPAIGNS  = int(os.getenv("SIM_N_CAMPAIGNS", "19"))

rng = np.random.default_rng(SIM_SEED)
OUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/simulated"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Scale down for simulation (proportional to real data) ─────────────────
# Real: 598k donors, 19 campaigns, 554k selection rows
# Simulated: 15k donors, 19 campaigns — enough to be realistic, fast to run
N_SEGS_PER_CAMP = 3       # ~3 segments per campaign = 57 action_ids
HIST_START      = pd.Timestamp("2011-02-01")
CAMP_START      = pd.Timestamp("2025-01-01")
CAMP_END        = pd.Timestamp("2026-02-03")

print("=" * 65)
print(f"BelFund Donor EV Scorer — Data Simulation")
print(f"Client: {CLIENT_NAME}  |  Seed: {SIM_SEED}")
print("=" * 65)
print(f"  Donors:    {N_DONORS:,}")
print(f"  Campaigns: {N_CAMPAIGNS}")
print(f"  Output:    {OUT_DIR.resolve()}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. DONOR POOL — ind_adr
# ═══════════════════════════════════════════════════════════════════════════
ind_ids = np.arange(100001, 100001 + N_DONORS)

# Language: Dutch 66%, French 34%, tiny English/German
lang = rng.choice(
    ["Dutch", "French", "English", "German"],
    size=N_DONORS, p=[0.659, 0.339, 0.001, 0.001]
)

# Gender: Female 46%, Male 27%, Couple 23%, Unknown 4%
gender = rng.choice(
    ["Female", "Male", "Couple", "Unknown"],
    size=N_DONORS, p=[0.459, 0.274, 0.225, 0.042]
)

# Province — use real proportions from profiling
provinces = [
    "Province of Antwerpen", "Province of Oost-Vlanderen",
    "Province of West-Vlanderen", "Province of Vlaams-Brabant",
    "Province of Hainaut", "Province of Liège",
    "Province of Limburg", "Brussels Capital",
    "Province of Namur", "Province of Brabant Wallon",
    "Unknown", "Province of Luxembourg"
]
prov_counts = [24904, 21461, 16963, 16772, 12146, 10658,
               9649,   9525,  5465,  4650,  3724,  2960]
prov_total  = sum(prov_counts)
prov_weights = [c / prov_total for c in prov_counts]
province = rng.choice(provinces, size=N_DONORS, p=prov_weights)

# District — map from province (simplified)
prov_to_district = {
    "Province of Antwerpen":      "District of Antwerpen",
    "Province of Oost-Vlanderen": "District of Gent",
    "Province of West-Vlanderen": "District of Brugge",
    "Province of Vlaams-Brabant": "District of Leuven",
    "Province of Hainaut":        "District of Mons",
    "Province of Liège":          "District of Liège",
    "Province of Limburg":        "District of Hasselt",
    "Brussels Capital":           "District of Bruxelles-Capitale",
    "Province of Namur":          "District of Namur",
    "Province of Brabant Wallon": "District of Nivelles",
    "Unknown":                    "Unknown",
    "Province of Luxembourg":     "District of Arlon",
}
district = [prov_to_district[p] for p in province]

ind_df = pd.DataFrame({
    "Client_Id":         CLIENT_ID,
    "Ind_id":            ind_ids,
    "Language_Original": lang,
    "Gender":            gender,
    "Province":          province,
    "District":          district,
    "zipcode":           rng.integers(1000, 9999, size=N_DONORS),
})
ind_df.to_csv(OUT_DIR / "ind_adr_sim.csv", index=False)
print(f"\n[1] ind_adr_sim: {ind_df.shape}")
print(f"    Language: {pd.Series(lang).value_counts().to_dict()}")
print(f"    Gender:   {pd.Series(gender).value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. CAMPAIGNS (actions_groups)
# Real: 19 in-scope FID campaigns, cost median €1.10, std €0.48
# ═══════════════════════════════════════════════════════════════════════════
# Spread campaigns chronologically across the training window
camp_dates = pd.date_range(CAMP_START, CAMP_END, periods=N_CAMPAIGNS)
action_group_ids = np.arange(9300001, 9300001 + N_CAMPAIGNS)

# CampaignSubType: Housemailing dominant (45%), Extrapolation 13%, Test 10%
subtypes = rng.choice(
    ["Housemailing", "Extrapolation", "Test", "Encartage", "NA"],
    size=N_CAMPAIGNS,
    p=[0.45, 0.13, 0.10, 0.06, 0.26]
)

# Cost: log-normal fit to real stats (mean=1.09, std=0.48)
# log-normal params: mu=ln(1.09)-0.5*ln(1+(0.48/1.09)^2) ≈ 0.017, sigma≈0.42
costs = np.random.lognormal(mean=0.017, sigma=0.42, size=N_CAMPAIGNS).clip(0.39, 4.94)

ag_rows = []
for i, (ag, cdate, sub, cost) in enumerate(zip(action_group_ids, camp_dates, subtypes, costs)):
    post_date = cdate + pd.Timedelta(days=7)
    ag_rows.append({
        "Client_Id":       CLIENT_ID,
        "Action_Group":    int(ag),
        "Campaign":        f"PELICANO_FID_DM_{cdate.strftime('%Y%m')}_{i+1:02d}",
        "CampaignChannel": "Direct Mail",
        "CampaignType":    "FID",
        "CampaignSubType": sub,
        "PostDate":        int(post_date.strftime("%Y%m%d")),
        "Cost_unit":       round(float(cost), 4),
    })

campaigns_df = pd.DataFrame(ag_rows)
campaigns_df.to_csv(OUT_DIR / "actions_groups_sim.csv", index=False)
print(f"\n[2] actions_groups_sim: {campaigns_df.shape}")
print(f"    Cost stats: mean=€{campaigns_df['Cost_unit'].mean():.2f}  "
      f"median=€{campaigns_df['Cost_unit'].median():.2f}  "
      f"std=€{campaigns_df['Cost_unit'].std():.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. SEGMENTS (actions)
# Real: 1,556 action_ids across all history; ~3 per campaign in scope
# ═══════════════════════════════════════════════════════════════════════════
seg_rows   = []
ai_counter = 9310001
action_id_to_ag = {}
action_id_to_cost = {}
action_id_to_date = {}

for ag_row in ag_rows:
    ag       = ag_row["Action_Group"]
    post_int = ag_row["PostDate"]
    post_dt  = pd.to_datetime(str(post_int), format="%Y%m%d")
    cost     = ag_row["Cost_unit"]
    for s in range(N_SEGS_PER_CAMP):
        ai = ai_counter
        ai_counter += 1
        action_id_to_ag[ai]   = ag
        action_id_to_cost[ai] = cost
        action_id_to_date[ai] = post_dt - pd.Timedelta(days=7)
        seg_rows.append({
            "Client_Id":    CLIENT_ID,
            "Action_Group": ag,
            "Action_id":    ai,
            "Segment":      f"SEG_{s+1:02d}",
            "PostDate":     post_int,
        })

segments_df = pd.DataFrame(seg_rows)
segments_df.to_csv(OUT_DIR / "actions_sim.csv", index=False)
all_action_ids = list(action_id_to_ag.keys())
print(f"\n[3] actions_sim: {segments_df.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. DONOR GIFT HISTORY
# Real stats:
#   - gifts per donor: median=2, mean=7.86, p75=6, p90=17, p99=102
#   - amount: log-normal mu=3.24, sigma=0.93 → geometric mean €25.58
#   - median gift €15, mean €39 (heavy right tail — outliers pulled mean up)
#   - 33% of donors are FID eligible (last gift < 730 days ago)
#   - donor tenure: median=1936 days, mean=2210
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n[4] Generating gift history for {N_DONORS:,} donors...")

donor_profiles = {}
gift_rows      = []
gift_id        = 5_000_001
HIST_DAYS      = (CAMP_END - HIST_START).days  # ~5479 days

# Simulate gift frequency: negative binomial calibrated to real (median=2, mean=7.86)
# NB(r=0.8, p=0.09) gives mean ~8, variance ~88 — close to real std=18.76
n_gifts_per_donor = rng.negative_binomial(0.8, 0.09, size=N_DONORS).clip(1, 336)

# FID eligibility: 33% of donors must have last gift < 730 days ago
# We achieve this by assigning recency directly then filling in history
fid_eligible = rng.random(N_DONORS) < 0.33

for idx, (ind, n_g, is_fid) in enumerate(zip(ind_ids, n_gifts_per_donor, fid_eligible)):
    # ── Gift dates ─────────────────────────────────────────────────────────
    if is_fid:
        # Last gift within 730 days of CAMP_END
        days_since_last = int(rng.integers(30, 729))
        last_gift_day   = HIST_DAYS - days_since_last
    else:
        # Last gift > 730 days ago (lapsed)
        days_since_last = int(rng.integers(730, 3900))
        last_gift_day   = max(0, HIST_DAYS - days_since_last)

    # Spread remaining gifts uniformly across history before last gift
    if n_g > 1:
        other_days = sorted(rng.integers(0, max(1, last_gift_day), size=n_g - 1).tolist())
        all_days   = other_days + [last_gift_day]
    else:
        all_days = [last_gift_day]

    gift_dates  = [HIST_START + pd.Timedelta(days=int(d)) for d in all_days]

    # ── Gift amounts — log-normal (mu=3.24, sigma=0.93) ────────────────────
    # Real: median=€15, geometric mean=€25.58
    # We use mu=2.71 (ln(15)) to hit the median correctly
    amounts = np.exp(rng.normal(2.71, 0.93, size=n_g)).clip(0.01, 900_000)

    donor_profiles[ind] = {
        "dates":        gift_dates,
        "amounts":      amounts,
        "n_gifts":      n_g,
        "mean_amount":  float(amounts.mean()),
        "last_gift_day": last_gift_day,
        "is_fid":       is_fid,
    }

    # ── Write gift rows ─────────────────────────────────────────────────────
    for gdate, amt in zip(gift_dates, amounts):
        gift_id += 1
        gift_rows.append({
            "Client_Id":          CLIENT_ID,
            "Gift_Transaction_Id": gift_id,
            "Ind_id":             int(ind),
            "Action_id":          None,
            "Action_Group":       None,
            "Amount":             round(float(amt), 2),
            "GiftStatus":         "Paid",
            "Pdate":              gdate.strftime("%Y-%m-%d"),
            "DateGiftCreated":    (gdate + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            "Flag_SDD":           rng.choice(["Y", None], p=[0.308, 0.692]),
            "OP":                 rng.choice(["Y", None], p=[0.063, 0.937]),
            "CampaignType":       "FID",
            "ContentType":        None,
            "GadgetType":         None,
        })

print(f"    Historical gift rows: {len(gift_rows):,}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. SELECTIONS + CAMPAIGN RESPONSE GIFTS
# Real: median 6036 donors per action_id; overall RR=6.53%
# RR by frequency band: 1gift→0%, 2-3→2.7%, 4-6→4.6%, 7-10→6.5%, 10+→13.2%
# RR by amount band: <€10→9.8%, €10-20→7.8%, €20-35→7.2%, €35-60→5.1%, >€60→6.2%
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n[5] Simulating selections and responses...")

sel_rows  = []
resp_gift_rows = []

# Each action_id selects ~40-60% of donors (gives median 6036/action like real)
# With 57 action_ids × ~6000 donors = ~342k selection rows
# For 15k donors that's ~22 selections per donor — matches real median of 8-20

for ai in all_action_ids:
    ag        = action_id_to_ag[ai]
    sel_date  = action_id_to_date[ai]
    cost      = action_id_to_cost[ai]
    sel_date_int = int(sel_date.strftime("%Y%m%d"))

    # Select 35-55% of donor pool per segment
    n_select  = int(rng.integers(int(N_DONORS * 0.35), int(N_DONORS * 0.55)))
    selected  = rng.choice(ind_ids, size=n_select, replace=False)

    for ind in selected:
        sel_rows.append({
            "Client_Id":    CLIENT_ID,
            "Action_Group": ag,
            "Action_id":    ai,
            "Ind_id":       int(ind),
            "Date":         sel_date_int,
        })

        # ── Response probability ────────────────────────────────────────────
        prof     = donor_profiles.get(int(ind))
        if prof is None:
            continue

        n_g   = prof["n_gifts"]
        m_amt = prof["mean_amount"]

        # Base RR by frequency band (from real data)
        if   n_g == 1:    base_rr = 0.000
        elif n_g <= 3:    base_rr = 0.027
        elif n_g <= 6:    base_rr = 0.046
        elif n_g <= 10:   base_rr = 0.065
        else:             base_rr = 0.132

        # Amount modifier (from real data)
        if   m_amt < 10:  amt_mod =  0.032
        elif m_amt < 20:  amt_mod =  0.012
        elif m_amt < 35:  amt_mod =  0.006
        elif m_amt < 60:  amt_mod = -0.015
        else:             amt_mod = -0.003

        # FID recency boost: recent donors more likely to respond
        rec_mod = 0.025 if prof["is_fid"] else -0.010

        p_resp  = float(np.clip(base_rr + amt_mod + rec_mod, 0.0, 0.40))

        if rng.random() < p_resp:
            # ── Simulated gift amount ─────────────────────────────────────
            # Real responder: median=€35, mean=€39, mu_log=3.24, sigma=0.93
            # Use real log-normal params directly, slight donor correlation
            donor_log_mean = np.log(max(m_amt, 1.0))
            amt = float(np.exp(rng.normal(3.56, 0.75)).clip(0.01, 5000))
            pdate = sel_date + pd.Timedelta(days=int(rng.integers(7, 45)))
            gift_id += 1
            resp_gift_rows.append({
                "Client_Id":          CLIENT_ID,
                "Gift_Transaction_Id": gift_id,
                "Ind_id":             int(ind),
                "Action_id":          int(ai),
                "Action_Group":       int(ag),
                "Amount":             round(amt, 2),
                "GiftStatus":         "Paid",
                "Pdate":              pdate.strftime("%Y-%m-%d"),
                "DateGiftCreated":    (pdate + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                "Flag_SDD":           rng.choice(["Y", None], p=[0.308, 0.692]),
                "OP":                 rng.choice(["Y", None], p=[0.063, 0.937]),
                "CampaignType":       "FID",
                "ContentType":        rng.choice(["None","FinancialAlert","FollowUp"], p=[0.75,0.15,0.10]),
                "GadgetType":         rng.choice(["NoGadget","SoftGadget","HardGadget"], p=[0.70,0.20,0.10]),
            })

# Write gifts
all_gift_rows = gift_rows + resp_gift_rows
gifts_df_sim = pd.DataFrame(all_gift_rows)
gifts_df_sim.to_csv(OUT_DIR / "gifts_valid_sim.csv", index=False)

# Write selections
selections_df_sim = pd.DataFrame(sel_rows)
selections_df_sim.to_csv(OUT_DIR / "tab_sel_sim.csv", index=False)

actual_rr = len(resp_gift_rows) / len(sel_rows) if sel_rows else 0
print(f"    Selection rows:     {len(sel_rows):,}")
print(f"    Response gifts:     {len(resp_gift_rows):,}")
print(f"    Actual RR:          {actual_rr:.2%}  (target: 6.53%)")
print(f"    Total gift rows:    {len(all_gift_rows):,}")

resp_amounts = pd.Series([r["Amount"] for r in resp_gift_rows])
if len(resp_amounts):
    print(f"\n    Amount stats (responders):")
    print(f"      mean=€{resp_amounts.mean():.2f}  median=€{resp_amounts.median():.2f}")
    print(f"      p90=€{resp_amounts.quantile(0.90):.2f}  p95=€{resp_amounts.quantile(0.95):.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. STANDING ORDERS (op)
# Real: 0.32% of selection pool, OpAct: 0=inactive(70%), 1=active(30%)
# ═══════════════════════════════════════════════════════════════════════════
n_op     = max(1, int(N_DONORS * 0.0032))
op_ids   = rng.choice(ind_ids, size=n_op, replace=False)
op_rows  = []
for ind in op_ids:
    op_rows.append({
        "Client_Id": CLIENT_ID,
        "Ind_id":    int(ind),
        "OpAct":     float(rng.choice([0, 1], p=[0.706, 0.294])),
        "Amount":    round(float(np.exp(rng.normal(2.5, 0.6)).clip(5, 200)), 2),
    })
op_df_sim = pd.DataFrame(op_rows)
op_df_sim.to_csv(OUT_DIR / "op_sim.csv", index=False)
print(f"\n[6] op_sim: {op_df_sim.shape}  ({n_op/N_DONORS:.2%} of pool)")

# ═══════════════════════════════════════════════════════════════════════════
# 7. SEPA DIRECT DEBITS (sdds)
# Real: 1.39% of pool, SddType: 5=Monthly(83%), 1=Yearly(11%), 2=2x/yr(3%)
# SddStatus: -1=cancelled(54%), 1=active(46%)
# SDD_amount: median ~€20 (from sample rows: 35,20,15,8,5)
# ═══════════════════════════════════════════════════════════════════════════
n_sdd    = max(1, int(N_DONORS * 0.0139))
sdd_ids  = rng.choice(ind_ids, size=n_sdd, replace=False)
sdd_type_map = {5:"Monthly", 1:"Yearly", 2:"2 times/Year", 0:"One-Off", 3:"3 times/Year", 4:"6 times/Year"}

sdd_rows = []
for ind in sdd_ids:
    sdd_type_code = int(rng.choice([5, 1, 2, 0, 3, 4],
                                    p=[0.830, 0.108, 0.030, 0.022, 0.008, 0.002]))
    sdd_status    = int(rng.choice([-1, 1], p=[0.542, 0.458]))
    sdd_amount    = round(float(np.exp(rng.normal(2.8, 0.7)).clip(5, 300)), 2)
    start_date    = CAMP_END - pd.Timedelta(days=int(rng.integers(30, 1000)))
    sdd_rows.append({
        "Client_Id":             CLIENT_ID,
        "Ind_id":                int(ind),
        "MandateRef":            f"0093{rng.integers(1e9,9e9)}",
        "SDD_amount":            sdd_amount,
        "recurring":             float(int(sdd_type_code > 0)),
        "ActionGroupSddOrigin":  int(rng.choice(action_group_ids)),
        "SDD_EndDate":           None,
        "Sdd_StartDate":         start_date.strftime("%Y-%m-%d"),
        "MandateRef_num":        float(rng.integers(int(9e11), int(9.4e11))),
        "SddType":               float(sdd_type_code),
        "SddStatus":             float(sdd_status),
    })

sdds_df_sim = pd.DataFrame(sdd_rows)
sdds_df_sim.to_csv(OUT_DIR / "sdds_sim.csv", index=False)
print(f"\n[7] sdds_sim: {sdds_df_sim.shape}  ({n_sdd/N_DONORS:.2%} of pool)")
print(f"    SddType: {pd.Series([r['SddType'] for r in sdd_rows]).value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════
# 8. IN-SCOPE SEGMENT FLAGS (PELICANO_FID_complete.xlsx equivalent)
# ═══════════════════════════════════════════════════════════════════════════
scope_rows = []
for ai in all_action_ids:
    ag   = action_id_to_ag[ai]
    pdate = action_id_to_date[ai] + pd.Timedelta(days=7)
    scope_rows.append({
        "action_id":             ai,
        "Action_group":          ag,
        "Post_Date":             pdate.strftime("%Y-%m-%d"),
        "In_scope":              "Y",
        "gadget":                rng.choice(["Y","N"], p=[0.20,0.80]),
        "Gadget (Soft/Hard)":    rng.choice(["S","H",None], p=[0.12,0.08,0.80]),
        "Financial alert":       rng.choice(["Y","N"], p=[0.15,0.85]),
        "Fiscal attest (exclude)": "N",
        "Follow up":             rng.choice(["Y","N"], p=[0.10,0.90]),
    })

scope_df_sim = pd.DataFrame(scope_rows)
scope_df_sim.to_excel(OUT_DIR / "PELICANO_FID_complete_sim.xlsx", index=False)
print(f"\n[8] PELICANO_FID_complete_sim.xlsx: {scope_df_sim.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# 9. VALIDATION — check simulated stats match real
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("VALIDATION — simulated vs real")
print("=" * 65)

sim_sel   = pd.DataFrame(sel_rows)
sim_gifts = pd.DataFrame(all_gift_rows)
sim_resp  = pd.DataFrame(resp_gift_rows) if resp_gift_rows else pd.DataFrame()

print(f"\nResponse rate:")
print(f"  Real:      6.53%")
print(f"  Simulated: {actual_rr:.2%}")

if not sim_resp.empty:
    ra = sim_resp["Amount"]
    print(f"\nAmount stats (responders):")
    print(f"  Real:      median=€35  mean=€39  p90=€65  p95=€100")
    print(f"  Simulated: median=€{ra.median():.0f}  mean=€{ra.mean():.0f}  "
          f"p90=€{ra.quantile(0.90):.0f}  p95=€{ra.quantile(0.95):.0f}")

sels_per_donor = sim_sel.groupby("Ind_id").size()
print(f"\nSelections per donor:")
print(f"  Real:      median=8  mean=20")
print(f"  Simulated: median={sels_per_donor.median():.0f}  mean={sels_per_donor.mean():.1f}")

hist_gifts = sim_gifts[sim_gifts["Action_id"].isna()]
gpd = hist_gifts.groupby("Ind_id").size()
print(f"\nGifts per donor (historical):")
print(f"  Real:      median=2  mean=7.86  p75=6  p90=17")
print(f"  Simulated: median={gpd.median():.0f}  mean={gpd.mean():.2f}  "
      f"p75={gpd.quantile(0.75):.0f}  p90={gpd.quantile(0.90):.0f}")

T_ref = CAMP_END
last_g = hist_gifts.groupby("Ind_id")["Pdate"].max()
last_g = pd.to_datetime(last_g)
dsl    = (T_ref - last_g).dt.days
print(f"\nDays since last gift (historical):")
print(f"  Real:      median=1420  FID eligible=33%")
print(f"  Simulated: median={dsl.median():.0f}  FID eligible={( dsl < 730).mean():.1%}")

print(f"\nCost per mail:")
print(f"  Real:      median=€1.10  mean=€1.09")
print(f"  Simulated: median=€{campaigns_df['Cost_unit'].median():.2f}  "
      f"mean=€{campaigns_df['Cost_unit'].mean():.2f}")

print("\n" + "=" * 65)
print(f"All files written to: {OUT_DIR.resolve()}")
print("=" * 65)
print("\nFiles:")
for f in sorted(OUT_DIR.glob("*")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size_kb:>8.1f} KB")
