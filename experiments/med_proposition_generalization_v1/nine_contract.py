import hashlib,json
CONTRACT='MEDPROP_SMALL4_CORE_FULL_SOURCE_V1'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
