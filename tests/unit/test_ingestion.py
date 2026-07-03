"""
Tests for the ingestion/parsing layer: the BILLSTATUS parser and subject
extraction. These are the pure functions that turn raw source data into the
bill dicts the rest of the pipeline consumes.
"""

from datetime import date

from precedent.preprocessing.entity_extraction import extract_subjects
from precedent.preprocessing.parsers.billstatus import parse_billstatus

# A minimal BILLSTATUS document in the modern (post-2022) schema, with an action
# history that should derive to "became_law".
MODERN_XML = """<?xml version="1.0"?>
<billStatus>
  <bill>
    <congress>118</congress>
    <type>HR</type>
    <number>1</number>
    <title>Test Enactment Act</title>
    <introducedDate>2023-01-09</introducedDate>
    <sponsors>
      <item>
        <bioguideId>A000001</bioguideId>
        <fullName>Rep. Ada Test</fullName>
        <party>D</party>
        <state>CA</state>
      </item>
    </sponsors>
    <cosponsors>
      <item><bioguideId>B000002</bioguideId><fullName>Rep. Bo Test</fullName>
        <party>R</party><state>TX</state></item>
    </cosponsors>
    <committees>
      <item><name>House Committee on Rules</name><chamber>House</chamber></item>
    </committees>
    <actions>
      <item><actionDate>2023-02-01</actionDate><text>Passed House</text><type>Floor</type></item>
      <item><actionDate>2023-03-01</actionDate><text>Passed Senate</text><type>Floor</type></item>
      <item><actionDate>2023-04-01</actionDate><text>Became Public Law No: 118-1</text><type>President</type></item>
    </actions>
    <latestAction><actionDate>2023-04-01</actionDate><text>Became Public Law</text></latestAction>
  </bill>
</billStatus>"""


def test_parse_billstatus_extracts_core_fields():
    bill = parse_billstatus(MODERN_XML)
    assert bill["congress"] == "118"
    assert bill["bill_type"] == "HR"
    assert bill["bill_number"] == "1"
    assert bill["title"] == "Test Enactment Act"
    assert bill["sponsor"]["bioguide_id"] == "A000001"
    assert len(bill["cosponsors"]) == 1
    assert bill["committees"][0]["name"] == "House Committee on Rules"
    assert len(bill["actions"]) == 3


def test_parse_billstatus_derives_became_law():
    # A Congress in the past so "ended" logic doesn't affect the became_law branch.
    bill = parse_billstatus(MODERN_XML, as_of=date(2025, 1, 1))
    assert bill["outcome"] == "became_law"


def test_parse_billstatus_bad_xml_returns_empty():
    assert parse_billstatus("<not-billstatus/>") == {}
    assert parse_billstatus("this is not xml <<<") == {}


def test_extract_subjects_derives_policy_areas():
    bill = {
        "title": "Prescription Drug Affordability Act",
        "summary": "Caps out-of-pocket prescription drug costs for Medicare enrollees "
        "and negotiates insurance prices for high-cost medications.",
    }
    subjects = extract_subjects(bill)
    assert "healthcare" in subjects


def test_extract_subjects_merges_existing():
    bill = {
        "title": "Tax bill",
        "summary": "revenue and tax credit changes",
        "subjects": ["custom-area"],
    }
    subjects = extract_subjects(bill)
    assert "custom-area" in subjects  # existing preserved
    assert "taxation" in subjects  # derived added
