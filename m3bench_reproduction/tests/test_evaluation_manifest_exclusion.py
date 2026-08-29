def test_exclusion_keeps_original_index():
 r={"global_index":4467,"evaluation_eligible":False,"exclusion_reason":"released_metadata_missing_gold"}
 assert r["global_index"]==4467 and r["exclusion_reason"]
