"""
Generate the seed dataset: a small, hand-curated set of bills with deliberate
structure so both engines have something interesting to work with.

Why generate it rather than hit the live GovInfo API for the demo:

* It runs offline in a second, so anyone can stand the app up instantly.
* It is *curated for the comparison*. Real random bills would rarely share
  sponsors or committees, so the graph would be sparse and the GraphRAG-vs-
  vector contrast would be invisible. Here, bills within a policy area share
  sponsors and committees on purpose, and a few sponsors deliberately cross
  policy areas, so the graph traversal has real edges to reason over -- and you
  can see where it reaches a precedent that pure text similarity would miss.

The shape produced matches ``precedent.models`` exactly, so these dicts flow
through the same enrich -> graph/vector loading path as real ingested bills.

Run:  python scripts/build_seed.py   (writes data/seed_bills.json)
"""

import json
from pathlib import Path

# --- shared entities, reused across bills to create graph connectivity ---

LEGISLATORS = {
    "H001": {"bioguide_id": "H001", "full_name": "Rep. Jordan Hayes", "party": "D", "state": "CA"},
    "W001": {"bioguide_id": "W001", "full_name": "Sen. Marcus Webb", "party": "R", "state": "TX"},
    "R001": {"bioguide_id": "R001", "full_name": "Rep. Elena Ruiz", "party": "D", "state": "NY"},
    "N001": {"bioguide_id": "N001", "full_name": "Sen. Priya Nair", "party": "D", "state": "WA"},
    "F001": {"bioguide_id": "F001", "full_name": "Rep. Tom Fletcher", "party": "R", "state": "OH"},
    "K001": {"bioguide_id": "K001", "full_name": "Sen. Grace Kim", "party": "R", "state": "FL"},
}

WAYS_MEANS = {"name": "House Committee on Ways and Means", "chamber": "House"}
SEN_FINANCE = {"name": "Senate Committee on Finance", "chamber": "Senate"}
ENERGY_COMMERCE = {"name": "House Committee on Energy and Commerce", "chamber": "House"}
SEN_HELP = {
    "name": "Senate Committee on Health, Education, Labor, and Pensions",
    "chamber": "Senate",
}
HOUSE_JUDICIARY = {"name": "House Committee on the Judiciary", "chamber": "House"}
SEN_JUDICIARY = {"name": "Senate Committee on the Judiciary", "chamber": "Senate"}
HOUSE_SCIENCE = {"name": "House Committee on Science, Space, and Technology", "chamber": "House"}


def bill(congress, btype, number, title, summary, sponsor, cosponsors, committees, outcome):
    return {
        "congress": congress,
        "bill_type": btype,
        "bill_number": number,
        "title": title,
        "introduced_date": f"{2021 if congress == '117' else 2023}-03-15",
        "sponsor": LEGISLATORS[sponsor],
        "cosponsors": [LEGISLATORS[c] for c in cosponsors],
        "committees": committees,
        "summary": summary,
        "outcome": outcome,
        "latest_action": {"date": "2024-01-10", "text": f"Outcome recorded: {outcome}."},
        "actions": [],
    }


BILLS = [
    # --- Healthcare cluster (Ruiz + Nair; Energy&Commerce, Senate HELP) ---
    bill(
        "118",
        "hr",
        "410",
        "Prescription Drug Affordability Act",
        "Caps out-of-pocket prescription drug costs for Medicare enrollees and lets the "
        "Secretary of Health negotiate prices for high-cost medications. Establishes an "
        "insurance rebate program to lower premiums. Extends coverage for chronic disease "
        "management.",
        "R001",
        ["N001"],
        [ENERGY_COMMERCE, SEN_HELP],
        "became_law",
    ),
    bill(
        "118",
        "s",
        "88",
        "Rural Hospital Preservation Act",
        "Provides grants to rural hospitals facing closure and funds telehealth "
        "infrastructure so patients in remote areas can access care. Increases Medicaid "
        "reimbursement rates for rural providers.",
        "N001",
        ["R001"],
        [SEN_HELP],
        "passed_one_chamber_pending",
    ),
    bill(
        "117",
        "hr",
        "1205",
        "Mental Health Access Expansion Act",
        "Expands insurance coverage for mental health and substance use treatment and funds "
        "community health clinics. Requires parity between mental and physical health "
        "benefits in employer plans.",
        "R001",
        ["N001", "H001"],
        [ENERGY_COMMERCE],
        "died_in_committee",
    ),
    bill(
        "117",
        "s",
        "640",
        "Insulin Cost Reduction Act",
        "Caps the monthly cost of insulin for insured patients and directs a study of drug "
        "pricing in the diabetes treatment market. Provides emergency insulin access grants.",
        "N001",
        ["R001"],
        [SEN_HELP, SEN_FINANCE],
        "died_after_passing_both_chambers",
    ),
    # --- Taxation cluster (Webb + Fletcher; Ways&Means, Senate Finance) ---
    bill(
        "118",
        "hr",
        "22",
        "Small Business Tax Relief Act",
        "Lowers the tax rate on small business income and expands the deduction for startup "
        "costs. Creates a tax credit for hiring workers in economically distressed areas.",
        "F001",
        ["W001"],
        [WAYS_MEANS],
        "became_law",
    ),
    bill(
        "118",
        "s",
        "150",
        "Corporate Tax Fairness Act",
        "Sets a minimum tax on large corporations and closes offshore revenue loopholes. "
        "Directs additional IRS enforcement funding toward high-income tax avoidance.",
        "W001",
        ["F001"],
        [SEN_FINANCE],
        "died_in_committee",
    ),
    bill(
        "117",
        "hr",
        "980",
        "Family Tax Credit Expansion Act",
        "Expands the child tax credit and makes it fully refundable for low-income families. "
        "Adds a dependent care deduction and indexes the credit to inflation.",
        "F001",
        ["W001", "R001"],
        [WAYS_MEANS],
        "vetoed",
    ),
    # --- Environment / energy cluster (Hayes + Nair; Energy&Commerce) ---
    bill(
        "118",
        "hr",
        "700",
        "Clean Energy Investment Act",
        "Provides tax credits for solar, wind, and battery storage projects and funds a grid "
        "modernization program to reduce emissions. Establishes clean energy workforce grants.",
        "H001",
        ["N001"],
        [ENERGY_COMMERCE, WAYS_MEANS],
        "passed_one_chamber_pending",
    ),
    bill(
        "118",
        "s",
        "300",
        "Carbon Emissions Reduction Act",
        "Sets national limits on carbon emissions from power plants and creates a market for "
        "emissions credits. Funds climate resilience projects in coastal communities.",
        "N001",
        ["H001"],
        [SEN_HELP],
        "died_in_committee",
    ),
    bill(
        "117",
        "hr",
        "455",
        "Electric Vehicle Infrastructure Act",
        "Funds a national network of electric vehicle charging stations and provides "
        "consumer rebates for clean vehicle purchases. Directs emissions studies of the "
        "transportation sector.",
        "H001",
        ["N001"],
        [ENERGY_COMMERCE],
        "became_law",
    ),
    # --- Technology / privacy cluster (Hayes + Kim; Science, Judiciary) ---
    bill(
        "118",
        "hr",
        "512",
        "Consumer Data Privacy Act",
        "Gives consumers the right to access and delete personal data held by companies and "
        "requires disclosure of data-sharing practices. Restricts sale of sensitive data "
        "without consent.",
        "H001",
        ["K001"],
        [HOUSE_JUDICIARY, HOUSE_SCIENCE],
        "died_in_committee",
    ),
    bill(
        "118",
        "s",
        "410",
        "Artificial Intelligence Accountability Act",
        "Requires audits of high-risk artificial intelligence systems and establishes "
        "transparency standards for automated decision-making. Directs a study of AI use in "
        "hiring and lending.",
        "K001",
        ["H001"],
        [SEN_JUDICIARY],
        "passed_one_chamber_pending",
    ),
    bill(
        "117",
        "hr",
        "888",
        "Children's Online Safety Act",
        "Restricts data collection from minors online and requires platforms to provide "
        "parental controls. Directs the FTC to enforce privacy protections for children.",
        "K001",
        ["H001"],
        [HOUSE_JUDICIARY],
        "became_law",
    ),
    # --- Immigration cluster (Webb + Ruiz; Judiciary) ---
    bill(
        "118",
        "hr",
        "150",
        "Border Security and Modernization Act",
        "Funds border surveillance technology and additional personnel while streamlining "
        "the visa processing system. Establishes penalties for visa overstays.",
        "W001",
        ["R001"],
        [HOUSE_JUDICIARY],
        "died_after_passing_both_chambers",
    ),
    bill(
        "118",
        "s",
        "222",
        "Dream Act Reauthorization",
        "Provides a path to citizenship for immigrants brought to the country as children "
        "and protects them from deportation. Expands access to in-state tuition and work "
        "authorization.",
        "R001",
        ["W001", "N001"],
        [SEN_JUDICIARY],
        "died_in_committee",
    ),
    bill(
        "117",
        "hr",
        "330",
        "Skilled Worker Visa Reform Act",
        "Increases the annual cap on skilled worker visas and creates a fast-track process "
        "for workers in high-demand technology fields. Removes per-country visa limits.",
        "R001",
        ["W001"],
        [HOUSE_JUDICIARY, HOUSE_SCIENCE],
        "passed_one_chamber_pending",
    ),
]


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "data" / "seed_bills.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(BILLS, f, indent=2)
    print(f"Wrote {len(BILLS)} seed bills to {out_path}")


if __name__ == "__main__":
    main()
