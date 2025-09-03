from ..apis.candid_api import CandidAPI

candid = CandidAPI(api_key="CANDID_API_KEY")

sample_response = {
  "code": 200,
  "message": "Request was processed successfully!",
  "took": 24,
  "time": "2020-01-01T01:01:01:01.000Z",
  "results_count": 2541,
  "page_count": 67,
  "errors": [
    "string"
  ],
  "hits": [
    {
      "organization": {
        "organization_id": "7578046",
        "ein": "39-1731296",
        "organization_name": "Candid",
        "also_known_as": "Foundation Center, Guidestar",
        "group_exemption": "3297",
        "mission": "Get you the information to do good",
        "website_url": "candid.org",
        "logo_url": "candid.org",
        "donation_page": "https://candid.org/about/funding-candid",
        "profile_level": "Platinum",
        "profile_year": 2020,
        "profile_link": "https://www.guidestar.org/profile/82-4267025",
        "profile_logo": "https://www.guidestar.org/App_Themes/MainSite2/images/ProfilePageSvgs/profile-PLATINUM2021-seal.svg",
        "leader_name": "Ann Mei Chang",
        "leader_title": "CEO",
        "contact_name": "John smith",
        "contact_email": "help@orgnam.org",
        "contact_phone": "(555) 111-5555",
        "contact_title": "Support lead",
        "number_of_employees": 55,
        "ruling_year": 2019
      },
      "properties": {
        "bmf_status": 'false',
        "pub78_verified": 'false',
        "allow_online_giving": 'true',
        "dei_submitted": 'false',
        "revoked": 'false',
        "defuncted_or_merged": 'false',
        "relationship_type": {
          "parent": 'true',
          "subordinate": 'false',
          "independent": 'false',
          "headquarters": 'true'
        }
      },
      "geography": {
        "address_line_1": "1 Financial Sq",
        "address_line_2": "Floor 24",
        "city": "New York",
        "state": "NY",
        "zip": 10005,
        "msa": "IL - Peoria-Pekin",
        "congressional_district": "District 45, CA",
        "county": "Peoria, IL",
        "latitude": 40.9052,
        "longitude": -89.5866
      },
      "taxonomies": {
        "subject_codes": [
          {
            "subject_code": "SP030000",
            "subject_code_description": "SP030000"
          }
        ],
        "population_served_codes": [
          {
            "population_served_code": "PG030000",
            "population_served_description": "People with Physical Disabilities  "
          }
        ],
        "ntee_codes": [
          {
            "ntee_code": "A00",
            "ntee_code_description": "Humanities"
          }
        ],
        "subsection_code": {
          "subsection_code": "03",
          "subsection_code_description": "501(c)(3) Public Charity"
        },
        "foundation_code": {
          "foundation_code": "15",
          "foundation_code_description": "50% tax deductible"
        }
      },
      "financials": {
        "most_recent_year": {
          "form_types": "990",
          "fiscal_year": 2020,
          "total_revenue": 2349999,
          "total_expenses": 22224499,
          "total_assets": 57426592
        },
        "bmf_gross_receipts": 2349999,
        "bmf_assets": 1849900,
        "required_to_file_990t": 'false',
        "a_133_audit_performed": 'false'
      },
      "dates": {
        "seal_last_modified": "2020-01-01T01:01:01:01.000Z",
        "profile_last_modified": "2020-01-01T01:01:01:01.000Z",
        "dei_last_modified": "2020-01-01T01:01:01:01.000Z",
        "financials_last_modified": "2020-01-01T01:01:01:01.000Z",
        "last_modified": "2020-01-01T01:01:01:01.000Z"
      }
    }
  ]
}

# Example categories to seed
categories = ["education", "animals", "health"]
# candid.seed_nonprofits(categories)


recordTrans = candid._transform_record(sample_response["hits"][0]) #["organization"])
print(recordTrans)

addRec = candid._add_single_nonprofit(sample_response["hits"][0]) #["organization"])
print(addRec)
print("Done")


