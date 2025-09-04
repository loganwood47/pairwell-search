from ..apis.candid_api import CandidEssentialsAPI
from .. import db

import json

import os

CANDID_ESSENTIALS_API_KEY = os.getenv("CANDID_ESSENTIALS_API_KEY")

candid = CandidEssentialsAPI(api_key=CANDID_ESSENTIALS_API_KEY)
# TODO: test single pull of real API using search terms

sample_response = {
    "code": 200,
    "message": "Request was processed successfully!",
    "took": 40,
    "time": "2024-02-12 18:46:45Z",
    "results_count": 6,
    "page_count": 1,
    "errors": [],
    "hits": [
        {
            "organization": {
                "organization_id": "8030190",
                "ein": "65-0988321",
                "organization_name": "Bonefish & Tarpon Trust",
                "also_known_as": "BTT",
                "group_exemption": "0000",
                "mission": "To conserve bonefish, tarpon, and permit--the species, their habitats and the fisheries they comprise through science-based conservation, education, and advocacy.",
                "website_url": "www.bonefishtarpontrust.org",
                "logo_url": "https://www.guidestar.org/ViewEdoc.aspx?eDocId=8718297&approved=True",
                "profile_level": "Platinum",
                "profile_year": 2023,
                "profile_link": "https://www.guidestar.org/profile/65-0988321",
                "profile_logo": "https://www.guidestar.org/App_Themes/MainSite2/images/ProfilePageSvgs/profile-PLATINUM2023-seal.svg",
                "leader_name": "Jim McDuffie",
                "leader_title": "President & CEO",
                "contact_name": "Jim Mcduffie",
                "contact_email": "jim@bonefishtarpontrust.org",
                "contact_phone": "(786) 618-9479",
                "contact_title": "President & CEO",
                "number_of_employees": "16",
                "ruling_year": 2001
            },
            "properties": {
                "bmf_status": True,
                "pub78_verified": True,
                "allow_online_giving": True,
                "dei_submitted": False,
                "revoked": False,
                "defunct_or_merged": False,
                "relationship_type": {
                    "parent": False,
                    "subordinate": False,
                    "independent": True,
                    "headquarters": False
                }
            },
            "geography": {
                "address_line_1": "2937 SW 27th Ave Ste 203",
                "address_line_2": None,
                "city": "Miami",
                "state": "FL",
                "zip": "33133",
                "msa": "FL - Miami",
                "congressional_district": "District 27, FL",
                "county": "Miami-dade, FL",
                "latitude": 25.7368,
                "longitude": -80.2377
            },
            "taxonomies": {
                "subject_codes": [
                    {
                        "subject_code": "SC060200",
                        "subject_code_description": "Nature education"
                    },
                    {
                        "subject_code": "SC060000",
                        "subject_code_description": "Environmental education"
                    },
                    {
                        "subject_code": "SC040102",
                        "subject_code_description": "Aquatic wildlife protection"
                    },
                    {
                        "subject_code": "SC040100",
                        "subject_code_description": "Wildlife biodiversity"
                    },
                    {
                        "subject_code": "SC040000",
                        "subject_code_description": "Biodiversity"
                    },
                    {
                        "subject_code": "SC030406",
                        "subject_code_description": "Wetlands"
                    },
                    {
                        "subject_code": "SC030403",
                        "subject_code_description": "Oceans and coastal waters"
                    },
                    {
                        "subject_code": "SC030400",
                        "subject_code_description": "Water resources"
                    },
                    {
                        "subject_code": "SC030000",
                        "subject_code_description": "Natural resources"
                    },
                    {
                        "subject_code": "SC000000",
                        "subject_code_description": "Environment"
                    }
                ],
                "population_served_codes": [
                    {
                        "population_served_code": "PA020000",
                        "population_served_description": "Adults"
                    },
                    {
                        "population_served_code": "PA010000",
                        "population_served_description": "Children and youth"
                    },
                    {
                        "population_served_code": "PA000000",
                        "population_served_description": "Age groups"
                    }
                ],
                "ntee_codes": [
                    {
                        "ntee_code": "D33",
                        "ntee_code_description": "Fisheries"
                    }
                ],
                "subsection_code": {
                    "subsection_code": "03",
                    "subsection_code_description": "501(c)(3) Public Charity"
                },
                "foundation_code": {
                    "foundation_code": "15",
                    "foundation_code_description": "Organization which receives a substantial part of its support from a governmental unit or the general public"
                }
            },
            "financials": {
                "most_recent_year": {
                    "form_type": "990",
                    "fiscal_year": 2022,
                    "total_revenue": 3835027.0,
                    "total_expenses": 3924905.0,
                    "total_assets": 2231762.0
                },
                "bmf_gross_receipts": 4444971.0,
                "bmf_assets": 2231762.0,
                "required_to_file_990t": True,
                "a_133_audit_performed": False
            },
            "dates": {
                "seal_last_modified": "2023-02-21T16:21:24.0000000",
                "profile_last_modified": "2023-02-21T16:24:32.0000000",
                "dei_last_modified": "2023-02-21T00:00:00.0000000",
                "financials_last_modified": "2023-12-22T05:01:43.0000000",
                "last_modified": "2023-12-22T05:01:43.0000000"
            }
        },
        {
            "organization": {
                "organization_id": "8962613",
                "ein": "27-3185735",
                "organization_name": "The CLEO Institute Inc",
                "also_known_as": None,
                "group_exemption": "0000",
                "mission": "TO EDUCATE & EMPOWER COMMUNITIES TO DEMAND CLIMATE ACTION, ENSURING A SAFE, JUST & HEALTHY ENVIRONMENT FOR ALL.",
                "website_url": "http://www.cleoinstitute.org",
                "logo_url": "https://www.guidestar.org/ViewEdoc.aspx?eDocId=9848307&approved=True",
                "profile_level": "Platinum",
                "profile_year": 2023,
                "profile_link": "https://www.guidestar.org/profile/27-3185735",
                "profile_logo": "https://www.guidestar.org/App_Themes/MainSite2/images/ProfilePageSvgs/profile-PLATINUM2023-seal.svg",
                "leader_name": "Yoca Arditi-Rocha",
                "leader_title": "Executive Director",
                "contact_name": "Olivia Collins",
                "contact_email": "olivia@cleoinstitute.org",
                "contact_phone": "(305) 573-5251",
                "contact_title": "Senior Director of Programs",
                "number_of_employees": "23",
                "ruling_year": 2010
            },
            "properties": {
                "bmf_status": True,
                "pub78_verified": True,
                "allow_online_giving": True,
                "dei_submitted": True,
                "revoked": False,
                "defunct_or_merged": False,
                "relationship_type": {
                    "parent": False,
                    "subordinate": False,
                    "independent": True,
                    "headquarters": False
                }
            },
            "geography": {
                "address_line_1": "2103 Coral Way Fl 2",
                "address_line_2": None,
                "city": "Miami",
                "state": "FL",
                "zip": "33145",
                "msa": "FL - Miami",
                "congressional_district": "District 27, FL",
                "county": "Miami-Dade, FL",
                "latitude": 25.7507,
                "longitude": -80.2281
            },
            "taxonomies": {
                "subject_codes": [
                    {
                        "subject_code": "SJ060200",
                        "subject_code_description": "Disaster preparedness"
                    },
                    {
                        "subject_code": "SJ060000",
                        "subject_code_description": "Disasters and emergency management"
                    },
                    {
                        "subject_code": "SJ000000",
                        "subject_code_description": "Public safety"
                    },
                    {
                        "subject_code": "SC000000",
                        "subject_code_description": "Environment"
                    },
                    {
                        "subject_code": "SB000000",
                        "subject_code_description": "Education"
                    }
                ],
                "population_served_codes": [
                    {
                        "population_served_code": "PC040000",
                        "population_served_description": "Women and girls"
                    },
                    {
                        "population_served_code": "PC000000",
                        "population_served_description": "Sexual identity"
                    },
                    {
                        "population_served_code": "PA020000",
                        "population_served_description": "Adults"
                    },
                    {
                        "population_served_code": "PA010000",
                        "population_served_description": "Children and youth"
                    },
                    {
                        "population_served_code": "PA000000",
                        "population_served_description": "Age groups"
                    }
                ],
                "ntee_codes": [
                    {
                        "ntee_code": "C60",
                        "ntee_code_description": "Environmental Education and Outdoor Survival Programs"
                    }
                ],
                "subsection_code": {
                    "subsection_code": "03",
                    "subsection_code_description": "501(c)(3) Public Charity"
                },
                "foundation_code": {
                    "foundation_code": "15",
                    "foundation_code_description": "Organization which receives a substantial part of its support from a governmental unit or the general public"
                }
            },
            "financials": {
                "most_recent_year": {
                    "form_type": "990",
                    "fiscal_year": 2022,
                    "total_revenue": 1625821.0,
                    "total_expenses": 1892577.0,
                    "total_assets": 874350.0
                },
                "bmf_gross_receipts": 1659429.0,
                "bmf_assets": 874350.0,
                "required_to_file_990t": False,
                "a_133_audit_performed": False
            },
            "dates": {
                "seal_last_modified": "2024-01-10T11:08:03.0000000",
                "profile_last_modified": "2024-01-10T11:14:44.0000000",
                "dei_last_modified": "2024-01-05T00:00:00.0000000",
                "financials_last_modified": "2023-12-22T05:01:43.0000000",
                "last_modified": "2024-01-10T11:14:44.0000000"
            }
        },
        {
            "organization": {
                "organization_id": "7041168",
                "ein": "20-5196010",
                "organization_name": "Dream In Green, Inc.",
                "also_known_as": "Green Schools Challenge",
                "group_exemption": "0000",
                "mission": "Dream in Green’s mission is to empower individuals, especially youth, to lead in the response to climate change and other environmental challenges facing South Florida..We assist diverse organizations, including schools, households, local governments and businesses to reduce their environmental footprint. Through establishing partnerships in our community, we develop, implement and oversee educational programs and workshops that promote environmentally sustainable behaviors among all age groups, with a particular emphasis on K-12 students.",
                "website_url": "www.dreamingreen.org",
                "logo_url": "https://www.guidestar.org/ViewEdoc.aspx?eDocId=9315985&approved=True",
                "profile_level": "Platinum",
                "profile_year": 2023,
                "profile_link": "https://www.guidestar.org/profile/20-5196010",
                "profile_logo": "https://www.guidestar.org/App_Themes/MainSite2/images/ProfilePageSvgs/profile-PLATINUM2023-seal.svg",
                "leader_name": "Barbara Martinez-Guerrero",
                "leader_title": None,
                "contact_name": "Barbara Martinez-Guerrero",
                "contact_email": "barbara@dreamingreen.org",
                "contact_phone": "(786) 574-4909",
                "contact_title": "Executive Director",
                "number_of_employees": "2",
                "ruling_year": 2006
            },
            "properties": {
                "bmf_status": True,
                "pub78_verified": True,
                "allow_online_giving": True,
                "dei_submitted": True,
                "revoked": False,
                "defunct_or_merged": False,
                "relationship_type": {
                    "parent": False,
                    "subordinate": False,
                    "independent": True,
                    "headquarters": False
                }
            },
            "geography": {
                "address_line_1": "2103 Coral Way Center For Social Change 2nd Floor",
                "address_line_2": None,
                "city": "Miami",
                "state": "FL",
                "zip": "33145",
                "msa": "FL - Miami",
                "congressional_district": "District 27, FL",
                "county": "Miami-Dade, FL",
                "latitude": 25.7537,
                "longitude": -80.2336
            },
            "taxonomies": {
                "subject_codes": [
                    {
                        "subject_code": "SC060000",
                        "subject_code_description": "Environmental education"
                    },
                    {
                        "subject_code": "SC030600",
                        "subject_code_description": "Energy resources"
                    },
                    {
                        "subject_code": "SC030000",
                        "subject_code_description": "Natural resources"
                    },
                    {
                        "subject_code": "SC000000",
                        "subject_code_description": "Environment"
                    },
                    {
                        "subject_code": "SB090000",
                        "subject_code_description": "Education services"
                    },
                    {
                        "subject_code": "SB000000",
                        "subject_code_description": "Education"
                    }
                ],
                "population_served_codes": [
                    {
                        "population_served_code": "PA020000",
                        "population_served_description": "Adults"
                    },
                    {
                        "population_served_code": "PA010000",
                        "population_served_description": "Children and youth"
                    },
                    {
                        "population_served_code": "PA000000",
                        "population_served_description": "Age groups"
                    }
                ],
                "ntee_codes": [
                    {
                        "ntee_code": "B90",
                        "ntee_code_description": "Educational Services and Schools - Other"
                    }
                ],
                "subsection_code": {
                    "subsection_code": "03",
                    "subsection_code_description": "501(c)(3) Public Charity"
                },
                "foundation_code": {
                    "foundation_code": "15",
                    "foundation_code_description": "Organization which receives a substantial part of its support from a governmental unit or the general public"
                }
            },
            "financials": {
                "most_recent_year": {
                    "form_type": "990",
                    "fiscal_year": 2022,
                    "total_revenue": 233201.0,
                    "total_expenses": 241212.0,
                    "total_assets": 261426.0
                },
                "bmf_gross_receipts": 233201.0,
                "bmf_assets": 261426.0,
                "required_to_file_990t": False,
                "a_133_audit_performed": False
            },
            "dates": {
                "seal_last_modified": "2023-08-02T11:02:35.0000000",
                "profile_last_modified": "2023-08-02T11:02:35.0000000",
                "dei_last_modified": "2020-06-19T16:55:50.0000000",
                "financials_last_modified": "2023-05-18T15:24:48.0000000",
                "last_modified": "2023-11-14T08:38:00.0000000"
            }
        },
        {
            "organization": {
                "organization_id": "8041374",
                "ein": "65-0350357",
                "organization_name": "The Miami Foundation",
                "also_known_as": None,
                "group_exemption": "0000",
                "mission": "The Miami Foundation builds the philanthropic, civic, and leadership backbone for Greater Miami. Since 1967, the Foundation has invested $485 million to strengthen our community with partnerships and contributions from more than 1000 fundholders and 35,000 donors. The Miami Foundation, which currently manages over $350 million in assets, mobilizes donors, nonprofits, leaders, and locals to set a bold vision for our community's future and to invest in a stronger, more equitable, more resilient Greater Miami.",
                "website_url": "http://www.miamifoundation.org/",
                "logo_url": "https://www.guidestar.org/ViewEdoc.aspx?eDocId=7139131&approved=True",
                "profile_level": "None",
                "profile_year":  None,
                "profile_link": "https://www.guidestar.org/profile/65-0350357",
                "profile_logo": None,
                "leader_name": "Rebecca Fishman Lipsey",
                "leader_title": "President and CEO",
                "contact_name": "Ms. Rebecca Fishman Lipsey",
                "contact_email": "rfl@miamifoundation.org",
                "contact_phone": None,
                "contact_title": "President and CEO",
                "number_of_employees": "50",
                "ruling_year": 1992
            },
            "properties": {
                "bmf_status": True,
                "pub78_verified": True,
                "allow_online_giving": True,
                "dei_submitted": False,
                "revoked": False,
                "defunct_or_merged": False,
                "relationship_type": {
                    "parent": False,
                    "subordinate": False,
                    "independent": True,
                    "headquarters": False
                }
            },
            "geography": {
                "address_line_1": "40 NW 3rd St Ste 305",
                "address_line_2": None,
                "city": "Miami",
                "state": "FL",
                "zip": "33128",
                "msa": "FL - Miami",
                "congressional_district": "District 24, FL",
                "county": "Miami-dade, FL",
                "latitude": 25.777,
                "longitude": -80.1944
            },
            "taxonomies": {
                "subject_codes": [
                    {
                        "subject_code": "SN030000",
                        "subject_code_description": "Community improvement"
                    },
                    {
                        "subject_code": "SN000000",
                        "subject_code_description": "Community and economic development"
                    },
                    {
                        "subject_code": "SD020000",
                        "subject_code_description": "Foundations"
                    },
                    {
                        "subject_code": "SD000000",
                        "subject_code_description": "Philanthropy"
                    },
                    {
                        "subject_code": "SA060000",
                        "subject_code_description": "Performing arts"
                    },
                    {
                        "subject_code": "SA000000",
                        "subject_code_description": "Arts and culture"
                    }
                ],
                "population_served_codes": [
                    {
                        "population_served_code": "PG030200",
                        "population_served_description": "Low-income people"
                    },
                    {
                        "population_served_code": "PG030000",
                        "population_served_description": "Economically disadvantaged people"
                    },
                    {
                        "population_served_code": "PG000000",
                        "population_served_description": "Social and economic status"
                    }
                ],
                "ntee_codes": [
                    {
                        "ntee_code": "T31",
                        "ntee_code_description": "Community Foundations"
                    }
                ],
                "subsection_code": {
                    "subsection_code": "03",
                    "subsection_code_description": "501(c)(3) Public Charity"
                },
                "foundation_code": {
                    "foundation_code": "15",
                    "foundation_code_description": "Organization which receives a substantial part of its support from a governmental unit or the general public"
                }
            },
            "financials": {
                "most_recent_year": {
                    "form_type": "990",
                    "fiscal_year": 2022,
                    "total_revenue": 113981496.0,
                    "total_expenses": 81136698.0,
                    "total_assets": 438006992.0
                },
                "bmf_gross_receipts": 279263466.0,
                "bmf_assets": 438006992.0,
                "required_to_file_990t": False,
                "a_133_audit_performed": True
            },
            "dates": {
                "seal_last_modified": "2023-09-28T19:00:12.0000000",
                "profile_last_modified": "2022-02-22T13:48:31.0000000",
                "dei_last_modified": "2021-01-20T11:59:17.0000000",
                "financials_last_modified": "2023-11-21T17:00:30.0000000",
                "last_modified": "2023-11-21T17:00:30.0000000"
            }
        },
        {
            "organization": {
                "organization_id": "9997334",
                "ein": "86-2827046",
                "organization_name": "Resilience Youth Network",
                "also_known_as": None,
                "group_exemption": "0000",
                "mission": None,
                "website_url": "resilienceyouthnetwork.org",
                "logo_url": None,
                "profile_level": "None",
                "profile_year":  None,
                "profile_link": "https://www.guidestar.org/profile/86-2827046",
                "profile_logo": None,
                "leader_name": None,
                "leader_title": None,
                "contact_name": None,
                "contact_email": None,
                "contact_phone": None,
                "contact_title": None,
                "number_of_employees": None,
                "ruling_year": 2021
            },
            "properties": {
                "bmf_status": True,
                "pub78_verified": True,
                "allow_online_giving": True,
                "dei_submitted": False,
                "revoked": False,
                "defunct_or_merged": False,
                "relationship_type": {
                    "parent": False,
                    "subordinate": False,
                    "independent": True,
                    "headquarters": False
                }
            },
            "geography": {
                "address_line_1": "441 NE 52nd St",
                "address_line_2": None,
                "city": "Miami",
                "state": "FL",
                "zip": "33137",
                "msa": "FL - Miami",
                "congressional_district": "District 24, FL",
                "county": "Miami-dade, FL",
                "latitude": 25.8237,
                "longitude": -80.186
            },
            "taxonomies": {
                "subject_codes": [
                    {
                        "subject_code": "SJ000000",
                        "subject_code_description": "Public safety"
                    }
                ],
                "population_served_codes": [],
                "ntee_codes": [
                    {
                        "ntee_code": "M01",
                        "ntee_code_description": "Alliance/Advocacy Organizations"
                    }
                ],
                "subsection_code": {
                    "subsection_code": "03",
                    "subsection_code_description": "501(c)(3) Public Charity"
                },
                "foundation_code": {
                    "foundation_code": "15",
                    "foundation_code_description": "Organization which receives a substantial part of its support from a governmental unit or the general public"
                }
            },
            "financials": {
                "most_recent_year": {
                    "form_type": None,
                    "fiscal_year":  None,
                    "total_revenue":  None,
                    "total_expenses":  None,
                    "total_assets":  None
                },
                "bmf_gross_receipts": 0.0,
                "bmf_assets": 0.0,
                "required_to_file_990t": False,
                "a_133_audit_performed": False
            },
            "dates": {
                "seal_last_modified": None,
                "profile_last_modified": None,
                "dei_last_modified": None,
                "financials_last_modified": None,
                "last_modified": "2022-09-09T13:27:18.0000000"
            }
        },
        {
            "organization": {
                "organization_id": "9469959",
                "ein": "47-3369805",
                "organization_name": "Resilience Action Fund Inc",
                "also_known_as": "RESILIENCE ACTION FUND",
                "group_exemption": "0000",
                "mission": "Empowerconsumerswithknowledgeandtoolstowithstandnaturalhazardswithlong-lastingandresilienthomesandcommunities,includingpublications,documentaries,buildingresilienceindex,homebuyers'guide,publicoutreachandothereducationaltools.",
                "website_url": "www.buildingresilient.com",
                "logo_url": None,
                "profile_level": "None",
                "profile_year":  None,
                "profile_link": "https://www.guidestar.org/profile/47-3369805",
                "profile_logo": None,
                "leader_name": None,
                "leader_title": None,
                "contact_name": None,
                "contact_email": None,
                "contact_phone": None,
                "contact_title": None,
                "number_of_employees": None,
                "ruling_year": 2015
            },
            "properties": {
                "bmf_status": True,
                "pub78_verified": True,
                "allow_online_giving": True,
                "dei_submitted": False,
                "revoked": False,
                "defunct_or_merged": False,
                "relationship_type": {
                    "parent": False,
                    "subordinate": False,
                    "independent": True,
                    "headquarters": False
                }
            },
            "geography": {
                "address_line_1": "441 NE 52nd St",
                "address_line_2": None,
                "city": "Miami",
                "state": "FL",
                "zip": "33137",
                "msa": "FL - Miami",
                "congressional_district": "District 24, FL",
                "county": "Miami-dade, FL",
                "latitude": 25.8237,
                "longitude": -80.186
            },
            "taxonomies": {
                "subject_codes": [
                    {
                        "subject_code": "SJ000000",
                        "subject_code_description": "Public safety"
                    }
                ],
                "population_served_codes": [],
                "ntee_codes": [
                    {
                        "ntee_code": "M01",
                        "ntee_code_description": "Alliance/Advocacy Organizations"
                    }
                ],
                "subsection_code": {
                    "subsection_code": "03",
                    "subsection_code_description": "501(c)(3) Public Charity"
                },
                "foundation_code": {
                    "foundation_code": "15",
                    "foundation_code_description": "Organization which receives a substantial part of its support from a governmental unit or the general public"
                }
            },
            "financials": {
                "most_recent_year": {
                    "form_type": "EZ",
                    "fiscal_year": 2022,
                    "total_revenue": 97776.0,
                    "total_expenses": 59012.0,
                    "total_assets": 400530.0
                },
                "bmf_gross_receipts": 97809.0,
                "bmf_assets": 400530.0,
                "required_to_file_990t": False,
                "a_133_audit_performed": False
            },
            "dates": {
                "seal_last_modified": None,
                "profile_last_modified": None,
                "dei_last_modified": None,
                "financials_last_modified": "2023-05-18T17:01:47.0000000",
                "last_modified": "2023-05-18T17:01:47.0000000"
            }
        }
    ]
}

json_response = json.dumps(sample_response)

print(json_response)
# Example categories to seed
categories = ["education", "animals", "health"]
# candid.seed_nonprofits(categories)


# recordTrans = candid._transform_record(sample_response["hits"][1]) #["organization"])
# print(recordTrans)
for hit in json.loads(json_response)['hits']:
    addRec = candid._add_single_nonprofit(hit)
    print(f"Added: {addRec} to DB")

    # first test
    # transObj = candid._transform_record(hit)
    # print(candid.check_nonprofit_exists_in_db(transObj["ein"]), transObj["ein"], transObj["name"])

# addRec = candid._add_single_nonprofit(sample_response["hits"][0]) #["organization"])
# print(addRec)
print("Done")


# print("TESTING DB QUERY")

# print(db.get_nonprofit_by_ein(ein="65-0988321"))