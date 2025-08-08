#!/usr/bin/env python3
"""
A script that finds 2 instances of a specific type in the same region and az and displays the cost per hour
"""
import argparse, json, sys, decimal, boto3
from botocore.config import Config
from botocore.exceptions import EndpointConnectionError

decimal.getcontext().prec = 9
DEBUG = False

def d(msg):
    if DEBUG:
        print(msg)

loc_map = {
    "us-east-1":  "US East (N. Virginia)",
    "us-east-2":  "US East (Ohio)",
    "us-west-1":  "US West (N. California)",
    "us-west-2":  "US West (Oregon)",
    "af-south-1": "Africa (Cape Town)",
    "ap-east-1":  "Asia Pacific (Hong Kong)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-south-2": "Asia Pacific (Hyderabad)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-southeast-3": "Asia Pacific (Jakarta)",
    "ap-southeast-4": "Asia Pacific (Melbourne)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ca-central-1":   "Canada (Central)",
    "eu-central-1":   "EU (Frankfurt)",
    "eu-central-2":   "EU (Zurich)",
    "eu-west-1":      "EU (Ireland)",
    "eu-west-2":      "EU (London)",
    "eu-west-3":      "EU (Paris)",
    "eu-south-1":     "EU (Milan)",
    "eu-south-2":     "EU (Spain)",
    "eu-north-1":     "EU (Stockholm)",
    "me-south-1":     "Middle East (Bahrain)",
    "me-central-1":   "Middle East (UAE)",
    "sa-east-1":      "South America (Sao Paulo)"
}

cfg = Config(retries={"max_attempts": 10, "mode": "adaptive"})

def pricing_client():
    for r in ("us-east-1", "eu-west-1", "ap-south-1"):
        try:
            d(f"pricing endpoint {r}")
            return boto3.client("pricing", region_name=r, config=cfg)
        except EndpointConnectionError:
            continue
    return boto3.client("pricing", region_name="us-east-1", config=cfg)

pricing     = pricing_client()
ec2_global  = boto3.client("ec2", config=cfg)

def location_from_catalog(region):
    pg = pricing.get_paginator("get_attribute_values")
    for p in pg.paginate(ServiceCode="AmazonEC2",
                         AttributeName="location",
                         MaxResults=100):
        for v in p["AttributeValues"]:
            if region in v["Value"]:
                return v["Value"]
    return None

def on_demand_price(instance, location):
    flt = [
        {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance},
        {"Type": "TERM_MATCH", "Field": "location",      "Value": location},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
    ]
    pg = pricing.get_paginator("get_products")
    for p in pg.paginate(ServiceCode="AmazonEC2", Filters=flt):
        for row in p["PriceList"]:
            data = json.loads(row)
            term = next(iter(data["terms"]["OnDemand"].values()))
            dim  = next(iter(term["priceDimensions"].values()))
            usd  = decimal.Decimal(dim["pricePerUnit"]["USD"])
            if usd != 0:
                return usd
    return None

def azs_for(region, instance):
    ec2 = boto3.client("ec2", region_name=region, config=cfg)
    azs = set()
    pg = ec2.get_paginator("describe_instance_type_offerings")
    for p in pg.paginate(LocationType="availability-zone",
                         Filters=[{"Name":"instance-type","Values":[instance]}]):
        azs.update(o["Location"] for o in p["InstanceTypeOfferings"])
    return azs

def cheapest_region(t1, t2):
    best = None
    for r in [x["RegionName"] for x in ec2_global.describe_regions()["Regions"]]:
        common = azs_for(r, t1) & azs_for(r, t2)
        if not common:
            continue
        loc = loc_map.get(r) or location_from_catalog(r)
        if not loc:
            continue
        p1, p2 = on_demand_price(t1, loc), on_demand_price(t2, loc)
        if p1 is None or p2 is None:
            continue
        combo = p1 + p2
        if best is None or combo < best["price"]:
            best = {"region": r, "price": combo, "azs": sorted(common)}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("type1")
    ap.add_argument("type2")
    args = ap.parse_args()

    res = cheapest_region(args.type1, args.type2)
    if res is None:
        print("no common region with valid non-zero pricing found", file=sys.stderr)
        sys.exit(1)

    letters = sorted({az[-1] for az in res["azs"]})
    print(f"Region: {res['region']}")
    print(f"AZ: {','.join(letters)}")
    print(f"CPH: {res['price']:.6f}")

if __name__ == "__main__":
    main()
