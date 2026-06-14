"""
Structured geographic metadata for every country/territory.

Used by:
  - CountryEliminationEngine (rule-based filtering)
  - BayesianGeoReasoner (priors, likelihoods, contradiction penalties)
  - RoadLineRules / UtilityPoleRules (country-specific infrastructure mappings)

Sources: ISO-3166, REST Countries, Wikipedia driving-side, Unicode script blocks.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Driving side
# ---------------------------------------------------------------------------

RIGHT_HAND_DRIVE_COUNTRIES: Set[str] = {
    "Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola",
    "Anguilla", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh",
    "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "British Indian Ocean Territory",
    "British Virgin Islands", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
    "Cambodia", "Cameroon", "Canada", "Cape Verde", "Caribbean Netherlands",
    "Cayman Islands", "Central African Republic", "Chad", "Chile", "China",
    "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros",
    "Cook Islands", "Costa Rica", "Croatia", "Cuba", "Curaçao", "Cyprus",
    "Czechia", "DR Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
    "Eswatini", "Ethiopia", "Falkland Islands", "Faroe Islands", "Fiji",
    "Finland", "France", "French Guiana", "French Polynesia",
    "French Southern and Antarctic Lands", "Gabon", "Gambia", "Georgia",
    "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", "Grenada",
    "Guadeloupe", "Guam", "Guatemala", "Guernsey", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hong Kong", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Isle of Man", "Israel", "Italy",
    "Ivory Coast", "Jamaica", "Japan", "Jersey", "Jordan", "Kazakhstan",
    "Kenya", "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia",
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania",
    "Luxembourg", "Macau", "Madagascar", "Malawi", "Malaysia", "Maldives",
    "Mali", "Malta", "Marshall Islands", "Martinique", "Mauritania", "Mauritius",
    "Mayotte", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia",
    "Montenegro", "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Caledonia", "New Zealand", "Nicaragua",
    "Niger", "Nigeria", "Niue", "Norfolk Island", "North Korea", "North Macedonia",
    "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Pitcairn Islands", "Poland", "Portugal", "Puerto Rico", "Qatar",
    "Republic of the Congo", "Romania", "Russia", "Rwanda", "Réunion",
    "Saint Barthélemy", "Saint Helena, Ascension and Tristan da Cunha",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin",
    "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Samoa",
    "San Marino", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Sint Maarten", "Slovakia", "Slovenia",
    "Solomon Islands", "Somalia", "South Africa", "South Georgia", "South Korea",
    "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname",
    "Svalbard and Jan Mayen", "Sweden", "Switzerland", "Syria",
    "São Tomé and Príncipe", "Taiwan", "Tajikistan", "Tanzania", "Thailand",
    "Timor-Leste", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia",
    "Turkey", "Turkmenistan", "Turks and Caicos Islands", "Tuvalu", "Uganda",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "United States Minor Outlying Islands", "United States Virgin Islands",
    "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam",
    "Wallis and Futuna", "Western Sahara", "Yemen", "Zambia", "Zimbabwe",
    "Åland Islands",
}

LEFT_HAND_DRIVE_COUNTRIES: Set[str] = {
    "United Kingdom", "Ireland", "Australia", "Japan", "India", "South Africa",
    "Thailand", "Indonesia", "Malaysia", "Singapore", "Hong Kong", "Macau",
    "New Zealand", "Bangladesh", "Pakistan", "Sri Lanka", "Nepal", "Bhutan",
    "Maldives", "Jamaica", "Bahamas", "Barbados", "Trinidad and Tobago",
    "Guyana", "Suriname", "Kenya", "Tanzania", "Uganda", "Zimbabwe", "Botswana",
    "Namibia", "Zambia", "Lesotho", "Eswatini", "Malawi", "Mozambique",
    "Cyprus", "Malta", "Brunei", "Fiji", "Kiribati", "Nauru", "Samoa",
    "Solomon Islands", "Tuvalu", "Tonga", "Papua New Guinea",
}

# Countries with ambiguous / mixed driving side (keep both)
MIXED_DRIVE_COUNTRIES: Set[str] = {
    "Hong Kong", "Macau", "Cyprus", "Malta", "Suriname",
}

# ---------------------------------------------------------------------------
# Scripts / writing systems
# ---------------------------------------------------------------------------

SCRIPT_TO_COUNTRIES: Dict[str, Set[str]] = {
    "latin": {
        "Albania", "Andorra", "Angola", "Argentina", "Australia", "Austria",
        "Bahamas", "Barbados", "Belgium", "Belize", "Benin", "Bolivia",
        "Bosnia and Herzegovina", "Botswana", "Brazil", "Bulgaria", "Burkina Faso",
        "Cameroon", "Canada", "Cape Verde", "Central African Republic", "Chad",
        "Chile", "Colombia", "Costa Rica", "Croatia", "Cuba", "Czechia",
        "DR Congo", "Denmark", "Dominica", "Dominican Republic", "Ecuador",
        "El Salvador", "Equatorial Guinea", "Estonia", "Eswatini", "Ethiopia",
        "Fiji", "Finland", "France", "Gabon", "Gambia", "Germany", "Ghana",
        "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
        "Haiti", "Honduras", "Hungary", "Iceland", "Indonesia", "Ireland",
        "Italy", "Ivory Coast", "Jamaica", "Latvia", "Lesotho", "Liberia",
        "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi",
        "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mexico",
        "Micronesia", "Moldova", "Monaco", "Montenegro", "Mozambique", "Namibia",
        "Nauru", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria",
        "Norway", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru",
        "Philippines", "Poland", "Portugal", "Puerto Rico", "Romania",
        "Samoa", "San Marino", "São Tomé and Príncipe", "Senegal", "Serbia",
        "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia",
        "Solomon Islands", "Somalia", "South Africa", "Spain", "Sudan",
        "Suriname", "Sweden", "Switzerland", "Tanzania", "Thailand", "Togo",
        "Tonga", "Trinidad and Tobago", "Turkey", "Tuvalu", "Uganda",
        "United Kingdom", "United States", "Uruguay", "Vanuatu", "Vatican City",
        "Venezuela", "Vietnam", "Zambia", "Zimbabwe", "Åland Islands",
    },
    "cyrillic": {
        "Belarus", "Bulgaria", "Kazakhstan", "Kyrgyzstan", "Moldova",
        "Mongolia", "Montenegro", "North Macedonia", "Russia", "Serbia",
        "Tajikistan", "Ukraine",
    },
    "arabic": {
        "Algeria", "Bahrain", "Chad", "Comoros", "Djibouti", "Egypt",
        "Eritrea", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Libya",
        "Mauritania", "Morocco", "Oman", "Palestine", "Qatar", "Saudi Arabia",
        "Somalia", "Sudan", "Syria", "Tunisia", "United Arab Emirates", "Yemen",
    },
    "chinese": {
        "China", "Hong Kong", "Macau", "Singapore", "Taiwan",
    },
    "devanagari": {
        "India", "Nepal",
    },
    "greek": {
        "Cyprus", "Greece",
    },
    "hebrew": {
        "Israel",
    },
    "japanese": {
        "Japan",
    },
    "korean": {
        "North Korea", "South Korea",
    },
    "thai": {
        "Thailand",
    },
    "armenian": {
        "Armenia",
    },
    "georgian": {
        "Georgia",
    },
    "ethiopic": {
        "Ethiopia", "Eritrea",
    },
}

# ---------------------------------------------------------------------------
# EU membership (for plate size / style rules)
# ---------------------------------------------------------------------------

EU_MEMBER_COUNTRIES: Set[str] = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
    "Slovenia", "Spain", "Sweden",
}

# ---------------------------------------------------------------------------
# Latitude bands (approximate, for vegetation / climate rules)
# ---------------------------------------------------------------------------

TROPICAL_COUNTRIES: Set[str] = {
    "Angola", "Antigua and Barbuda", "Bahamas", "Bangladesh", "Barbados",
    "Belize", "Benin", "Bolivia", "Brazil", "Brunei", "Burkina Faso",
    "Burundi", "Cambodia", "Cameroon", "Central African Republic", "Chad",
    "Colombia", "Comoros", "Costa Rica", "Cuba", "DR Congo", "Djibouti",
    "Dominica", "Dominican Republic", "Ecuador", "El Salvador", "Equatorial Guinea",
    "Eritrea", "Ethiopia", "Fiji", "French Guiana", "Gabon", "Gambia", "Ghana",
    "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
    "Honduras", "India", "Indonesia", "Ivory Coast", "Jamaica", "Kenya",
    "Kiribati", "Laos", "Liberia", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Mozambique", "Myanmar", "Namibia", "Nauru",
    "Nicaragua", "Niger", "Nigeria", "Palau", "Panama", "Papua New Guinea",
    "Paraguay", "Peru", "Philippines", "Republic of the Congo", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
    "Samoa", "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone",
    "Singapore", "Solomon Islands", "Somalia", "Sri Lanka", "Sudan", "Suriname",
    "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago",
    "Tuvalu", "Uganda", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia",
    "Zimbabwe",
}

SUBTROPICAL_COUNTRIES: Set[str] = {
    "Algeria", "Argentina", "Australia", "Bahrain", "Bhutan", "Botswana",
    "Chile", "China", "Egypt", "Eswatini", "Georgia", "Hong Kong", "Iran",
    "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait", "Lebanon",
    "Lesotho", "Libya", "Macau", "Morocco", "Nepal", "New Caledonia",
    "North Korea", "Oman", "Pakistan", "Palestine", "Qatar", "Saudi Arabia",
    "South Africa", "South Korea", "Syria", "Taiwan", "Tunisia",
    "Turkey", "Turkmenistan", "United Arab Emirates", "United States",
    "Uruguay", "Uzbekistan", "Western Sahara",
}

TEMPERATE_COUNTRIES: Set[str] = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Canada", "Croatia", "Cyprus",
    "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany",
    "Gibraltar", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
    "Kosovo", "Kyrgyzstan", "Latvia", "Liechtenstein", "Lithuania",
    "Luxembourg", "Malta", "Moldova", "Monaco", "Mongolia", "Montenegro",
    "Netherlands", "New Zealand", "North Macedonia", "Norway", "Poland",
    "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia",
    "Slovenia", "Spain", "Sweden", "Switzerland", "Tajikistan", "Ukraine",
    "United Kingdom", "Vatican City", "Åland Islands",
}

ARCTIC_ALPINE_COUNTRIES: Set[str] = {
    "Antarctica", "Faroe Islands", "Greenland", "Iceland", "Norway",
    "Svalbard and Jan Mayen",
}

# ---------------------------------------------------------------------------
# Climate / biome proxies
# ---------------------------------------------------------------------------

MEDITERRANEAN_CLIMATE_COUNTRIES: Set[str] = {
    "Albania", "Algeria", "Bosnia and Herzegovina", "Croatia", "Cyprus",
    "France", "Gibraltar", "Greece", "Israel", "Italy", "Lebanon", "Libya",
    "Malta", "Monaco", "Montenegro", "Morocco", "North Macedonia", "Palestine",
    "Portugal", "Slovenia", "Spain", "Syria", "Tunisia", "Turkey",
    "Western Sahara",
}

DESERT_COUNTRIES: Set[str] = {
    "Algeria", "Bahrain", "Chad", "Egypt", "Eritrea", "Iran", "Iraq",
    "Israel", "Jordan", "Kuwait", "Libya", "Mauritania", "Mongolia",
    "Morocco", "Namibia", "Niger", "Oman", "Qatar", "Saudi Arabia",
    "Sudan", "Syria", "Tunisia", "Turkmenistan", "United Arab Emirates",
    "Western Sahara", "Yemen",
}

# ---------------------------------------------------------------------------
# Utility pole types by country (elite GeoGuessr signal)
# ---------------------------------------------------------------------------

UTILITY_POLE_TYPE_COUNTRIES: Dict[str, Set[str]] = {
    "wooden_crossarm_us_style": {
        "United States", "Canada", "Philippines", "Japan", "South Korea",
        "Taiwan", "Thailand",
    },
    "wooden_triangle_crossarm": {
        "United Kingdom", "Ireland", "Australia", "New Zealand", "South Africa",
        "India", "Pakistan", "Bangladesh", "Sri Lanka", "Malaysia", "Singapore",
    },
    "concrete_spindle_europe": {
        "Germany", "France", "Netherlands", "Belgium", "Austria", "Switzerland",
        "Czechia", "Slovakia", "Poland", "Hungary", "Denmark", "Sweden",
        "Norway", "Finland", "Estonia", "Latvia", "Lithuania",
    },
    "concrete_tubular_mediterranean": {
        "Spain", "Portugal", "Italy", "Greece", "Croatia", "Slovenia",
        "Albania", "Montenegro", "Bosnia and Herzegovina", "Serbia",
        "North Macedonia", "Bulgaria", "Romania", "Cyprus", "Malta",
        "Turkey", "Israel", "Lebanon", "Tunisia", "Algeria", "Morocco",
    },
    "steel_lattice_high_tension": {
        "Russia", "Ukraine", "Belarus", "Kazakhstan", "China", "India",
        "Brazil", "Argentina", "South Africa", "Australia", "United States",
        "Canada", "Germany", "France", "United Kingdom",
    },
    "steel_tubular_modern": {
        "Japan", "South Korea", "Singapore", "United Arab Emirates", "Qatar",
        "Saudi Arabia", "Kuwait", "Bahrain", "Oman", "China", "Hong Kong",
        "Taiwan", "Malaysia", "Thailand", "Indonesia",
    },
    "wooden_pole_no_crossarm_single_wire": {
        "Norway", "Sweden", "Finland", "Estonia", "Latvia", "Iceland",
        "New Zealand", "Chile", "Argentina",
    },
}

# ---------------------------------------------------------------------------
# Road line / marking styles by country
# ---------------------------------------------------------------------------

ROAD_MARKING_STYLE_COUNTRIES: Dict[str, Set[str]] = {
    "yellow_center_white_edge_us": {
        "United States", "Canada", "Mexico", "Argentina", "Chile", "Brazil",
        "Colombia", "Peru", "Venezuela", "Ecuador", "Bolivia", "Paraguay",
        "Uruguay",
    },
    "white_center_white_edge_europe": {
        "United Kingdom", "Ireland", "France", "Germany", "Italy", "Spain",
        "Portugal", "Netherlands", "Belgium", "Switzerland", "Austria",
        "Sweden", "Norway", "Denmark", "Finland", "Poland", "Czechia",
        "Slovakia", "Hungary", "Romania", "Bulgaria", "Croatia", "Slovenia",
        "Greece", "Cyprus", "Malta", "Estonia", "Latvia", "Lithuania",
        "Luxembourg", "Liechtenstein", "Monaco", "Andorra", "Iceland",
        "Åland Islands", "Faroe Islands", "Gibraltar",
    },
    "yellow_double_solid_no_passing": {
        "United States", "Canada", "Australia", "New Zealand", "Japan",
        "South Korea", "Taiwan", "Thailand", "Malaysia", "Singapore",
    },
    "dashed_white_eu_motorway": {
        "Germany", "France", "Netherlands", "Belgium", "Austria", "Switzerland",
        "Italy", "Spain", "Portugal", "Czechia", "Slovakia", "Poland",
        "Hungary", "Slovenia", "Croatia", "Denmark", "Sweden", "Norway",
        "Finland",
    },
    "yellow_center_russia_cis": {
        "Russia", "Ukraine", "Belarus", "Kazakhstan", "Kyrgyzstan",
        "Tajikistan", "Uzbekistan", "Moldova", "Armenia", "Azerbaijan",
        "Georgia",
    },
    "white_dashed_center_uk_commonwealth": {
        "United Kingdom", "Ireland", "Australia", "New Zealand", "India",
        "Pakistan", "South Africa", "Kenya", "Nigeria", "Ghana", "Malaysia",
        "Singapore", "Hong Kong", "Jamaica", "Barbados", "Trinidad and Tobago",
    },
    "no_center_line_rural": {
        "Norway", "Sweden", "Finland", "Iceland", "Estonia", "Latvia",
        "New Zealand", "Chile", "Argentina", "Uruguay",
    },
    "red_white_curb_asia": {
        "Japan", "South Korea", "Taiwan", "Thailand", "Singapore", "Hong Kong",
        "Macau", "Malaysia", "Indonesia", "Philippines",
    },
    "yellow_curb_no_parking_eu": {
        "France", "Spain", "Italy", "Portugal", "Belgium", "Netherlands",
        "Germany", "Austria", "Switzerland", "Greece", "Cyprus", "Malta",
        "Croatia", "Slovenia", "Czechia", "Slovakia", "Hungary", "Poland",
        "Romania", "Bulgaria",
    },
}

# ---------------------------------------------------------------------------
# License plate region groups
# ---------------------------------------------------------------------------

EUROPEAN_PLATE_COUNTRIES: Set[str] = {
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland",
    "France", "Germany", "Gibraltar", "Greece", "Hungary", "Iceland", "Ireland",
    "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
    "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia",
    "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine",
    "United Kingdom", "Vatican City", "Åland Islands", "Faroe Islands",
}

NORTH_AMERICAN_PLATE_COUNTRIES: Set[str] = {
    "United States", "Canada", "Mexico",
}

# ---------------------------------------------------------------------------
# Helper lookups
# ---------------------------------------------------------------------------

ALL_COUNTRIES: Set[str] = set()
for s in (RIGHT_HAND_DRIVE_COUNTRIES, LEFT_HAND_DRIVE_COUNTRIES):
    ALL_COUNTRIES.update(s)


def get_country_set(criterion: str, value: str) -> Set[str]:
    """Generic lookup: e.g. get_country_set('script', 'cyrillic')."""
    if criterion == "script":
        return SCRIPT_TO_COUNTRIES.get(value, set())
    if criterion == "drive_side":
        if value == "left":
            return LEFT_HAND_DRIVE_COUNTRIES
        if value == "right":
            return RIGHT_HAND_DRIVE_COUNTRIES
        return set()
    if criterion == "climate":
        if value == "tropical":
            return TROPICAL_COUNTRIES
        if value == "subtropical":
            return SUBTROPICAL_COUNTRIES
        if value == "temperate":
            return TEMPERATE_COUNTRIES
        if value == "mediterranean":
            return MEDITERRANEAN_CLIMATE_COUNTRIES
        if value == "desert":
            return DESERT_COUNTRIES
        return set()
    if criterion == "eu_member":
        return EU_MEMBER_COUNTRIES if value == "true" else ALL_COUNTRIES - EU_MEMBER_COUNTRIES
    if criterion == "pole_type":
        return UTILITY_POLE_TYPE_COUNTRIES.get(value, set())
    if criterion == "road_marking":
        return ROAD_MARKING_STYLE_COUNTRIES.get(value, set())
    return set()


def country_code_to_name(code: str) -> str:
    """Best-effort ISO-3166 alpha-2 to country name (common codes only)."""
    mapping = {
        "AF": "Afghanistan", "AL": "Albania", "DZ": "Algeria", "AD": "Andorra",
        "AO": "Angola", "AR": "Argentina", "AM": "Armenia", "AU": "Australia",
        "AT": "Austria", "AZ": "Azerbaijan", "BS": "Bahamas", "BH": "Bahrain",
        "BD": "Bangladesh", "BY": "Belarus", "BE": "Belgium", "BZ": "Belize",
        "BJ": "Benin", "BT": "Bhutan", "BO": "Bolivia", "BA": "Bosnia and Herzegovina",
        "BW": "Botswana", "BR": "Brazil", "BN": "Brunei", "BG": "Bulgaria",
        "BF": "Burkina Faso", "BI": "Burundi", "KH": "Cambodia", "CM": "Cameroon",
        "CA": "Canada", "CV": "Cape Verde", "CF": "Central African Republic",
        "TD": "Chad", "CL": "Chile", "CN": "China", "CO": "Colombia",
        "KM": "Comoros", "CR": "Costa Rica", "HR": "Croatia", "CU": "Cuba",
        "CY": "Cyprus", "CZ": "Czechia", "CD": "DR Congo", "DK": "Denmark",
        "DJ": "Djibouti", "DM": "Dominica", "DO": "Dominican Republic",
        "EC": "Ecuador", "EG": "Egypt", "SV": "El Salvador", "GQ": "Equatorial Guinea",
        "ER": "Eritrea", "EE": "Estonia", "SZ": "Eswatini", "ET": "Ethiopia",
        "FJ": "Fiji", "FI": "Finland", "FR": "France", "GA": "Gabon",
        "GM": "Gambia", "GE": "Georgia", "DE": "Germany", "GH": "Ghana",
        "GR": "Greece", "GT": "Guatemala", "GN": "Guinea", "GW": "Guinea-Bissau",
        "GY": "Guyana", "HT": "Haiti", "HN": "Honduras", "HK": "Hong Kong",
        "HU": "Hungary", "IS": "Iceland", "IN": "India", "ID": "Indonesia",
        "IR": "Iran", "IQ": "Iraq", "IE": "Ireland", "IL": "Israel",
        "IT": "Italy", "JM": "Jamaica", "JP": "Japan", "JO": "Jordan",
        "KZ": "Kazakhstan", "KE": "Kenya", "KI": "Kiribati", "KW": "Kuwait",
        "KG": "Kyrgyzstan", "LA": "Laos", "LV": "Latvia", "LB": "Lebanon",
        "LS": "Lesotho", "LR": "Liberia", "LY": "Libya", "LI": "Liechtenstein",
        "LT": "Lithuania", "LU": "Luxembourg", "MO": "Macau", "MG": "Madagascar",
        "MW": "Malawi", "MY": "Malaysia", "MV": "Maldives", "ML": "Mali",
        "MT": "Malta", "MH": "Marshall Islands", "MR": "Mauritania",
        "MU": "Mauritius", "MX": "Mexico", "FM": "Micronesia", "MD": "Moldova",
        "MC": "Monaco", "MN": "Mongolia", "ME": "Montenegro", "MA": "Morocco",
        "MZ": "Mozambique", "MM": "Myanmar", "NA": "Namibia", "NR": "Nauru",
        "NP": "Nepal", "NL": "Netherlands", "NZ": "New Zealand", "NI": "Nicaragua",
        "NE": "Niger", "NG": "Nigeria", "KP": "North Korea", "MK": "North Macedonia",
        "NO": "Norway", "OM": "Oman", "PK": "Pakistan", "PW": "Palau",
        "PS": "Palestine", "PA": "Panama", "PG": "Papua New Guinea",
        "PY": "Paraguay", "PE": "Peru", "PH": "Philippines", "PL": "Poland",
        "PT": "Portugal", "QA": "Qatar", "CG": "Republic of the Congo",
        "RO": "Romania", "RU": "Russia", "RW": "Rwanda", "KN": "Saint Kitts and Nevis",
        "LC": "Saint Lucia", "VC": "Saint Vincent and the Grenadines",
        "WS": "Samoa", "SM": "San Marino", "ST": "São Tomé and Príncipe",
        "SA": "Saudi Arabia", "SN": "Senegal", "RS": "Serbia", "SC": "Seychelles",
        "SL": "Sierra Leone", "SG": "Singapore", "SK": "Slovakia", "SI": "Slovenia",
        "SB": "Solomon Islands", "SO": "Somalia", "ZA": "South Africa",
        "KR": "South Korea", "SS": "South Sudan", "ES": "Spain", "LK": "Sri Lanka",
        "SD": "Sudan", "SR": "Suriname", "SE": "Sweden", "CH": "Switzerland",
        "SY": "Syria", "TW": "Taiwan", "TJ": "Tajikistan", "TZ": "Tanzania",
        "TH": "Thailand", "TL": "Timor-Leste", "TG": "Togo", "TO": "Tonga",
        "TT": "Trinidad and Tobago", "TN": "Tunisia", "TR": "Turkey",
        "TM": "Turkmenistan", "TV": "Tuvalu", "UG": "Uganda", "UA": "Ukraine",
        "AE": "United Arab Emirates", "GB": "United Kingdom", "US": "United States",
        "UY": "Uruguay", "UZ": "Uzbekistan", "VU": "Vanuatu", "VA": "Vatican City",
        "VE": "Venezuela", "VN": "Vietnam", "YE": "Yemen", "ZM": "Zambia",
        "ZW": "Zimbabwe",
    }
    return mapping.get(code.upper(), "")

