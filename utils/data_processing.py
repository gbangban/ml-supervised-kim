import pandas as pd
import numpy as np 
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

null_strings = ['(null)','#N/A', 'NA', '?', '', ' ', '&&', 'nan']


categorical_columns_to_be_encoded = ['STOP_WAS_INITIATED']

column_categories = {
    'datetime': ['STOP_FRISK_DATE'],
    'numeric': ['YEAR2', 'STOP_ID', 'OBSERVED_DURATION_MINUTES', 'STOP_DURATION_MINUTES', 'SUSPECT_REPORTED_AGE',	'SUSPECT_WEIGHT', 'STOP_LOCATION_X',
                'STOP_LOCATION_Y'],
    'string': ['MONTH2', 'STOP_WAS_INITIATED', 'ISSUING_OFFICER_RANK', 'SUPERVISING_OFFICER_RANK', 'JURISDICTION_DESCRIPTION', 'SUSPECT_ARREST_OFFENSE',
               'SUMMONS_OFFENSE_DESCRIPTION', 'SUSPECTED_CRIME_DESCRIPTION',  'DEMEANOR_OF_PERSON_STOPPED', 'SUSPECT_SEX', 
               'SUSPECT_RACE_DESCRIPTION', 'SUSPECT_BODY_BUILD_TYPE', 'SUSPECT_OTHER_DESCRIPTION','STOP_LOCATION_BORO_NAME'
    ],
    'boolean': ['OFFICER_EXPLAINED_STOP_FLAG', 'OTHER_PERSON_STOPPED_FLAG', 'SUSPECT_ARRESTED_FLAG',
               'SUMMONS_ISSUED_FLAG', 'OFFICER_IN_UNIFORM_FLAG', 'FRISKED_FLAG',
               'SEARCHED_FLAG', 'ASK_FOR_CONSENT_FLG', 'CONSENT_GIVEN_FLG', 'OTHER_CONTRABAND_FLAG', 'FIREARM_FLAG', 'KNIFE_CUTTER_FLAG',
               'OTHER_WEAPON_FLAG', 'WEAPON_FOUND_FLAG', 'PHYSICAL_FORCE_CEW_FLAG', 'PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG', 'PHYSICAL_FORCE_HANDCUFF_SUSPECT_FLAG',
               'PHYSICAL_FORCE_OC_SPRAY_USED_FLAG', 'PHYSICAL_FORCE_OTHER_FLAG', 'PHYSICAL_FORCE_RESTRAINT_USED_FLAG', 'PHYSICAL_FORCE_VERBAL_INSTRUCTION_FLAG',
               'PHYSICAL_FORCE_WEAPON_IMPACT_FLAG', 'SEARCH_BASIS_ADMISSION_FLAG',	'SEARCH_BASIS_CONSENT_FLAG',	'SEARCH_BASIS_HARD_OBJECT_FLAG',	
               'SEARCH_BASIS_INCIDENTAL_TO_ARREST_FLAG', 'SEARCH_BASIS_OTHER_FLAG',	'SEARCH_BASIS_OUTLINE_FLAG'],
    # One offs/unique cols
    'height': ['SUSPECT_HEIGHT'],
    'hours': ['STOP_FRISK_TIME'],
}
columns_to_keep = [column for category in column_categories.values() for column in category] 


def clean_numeric(df, cols):
    for col in cols:
        df[col] = (
            df[col]
            # .replace(null_strings, np.nan)
            .apply(pd.to_numeric, errors='coerce')  # Coerce non-numeric to NaN
            .astype('Int64')  # Convert to nullable integer
        )
    return df

def clean_string(df, cols):
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .replace(null_strings, np.nan)
            .str.strip()  # Remove leading/trailing whitespace
        )
    return df

def clean_boolean(df, cols):
    pd.set_option('future.no_silent_downcasting', True) # Future proofing
    for col in cols:
        value_counts = df[col].value_counts()
        # These are boolean fields, so we shouldn't see anything other than two options, whether null or false, but never both.
        if len(value_counts) < 2:  # Force features with only 'Y' values to have 'N' values instead of nulls
            df[col] = df[col].replace([np.nan], 'N')
        df[col] = (
            df[col]
            # .replace(null_strings, False)
            .str.strip()  # Remove leading/trailing whitespace
            .replace({'Y': True, 'N': False, '1': True, '0': False}).infer_objects(copy=False)
            .astype('boolean')  # Pandas' nullable boolean
        )
        num_null_values = df[col].isna().sum()
        assert num_null_values >= 0 if col in ['ASK_FOR_CONSENT_FLG','CONSENT_GIVEN_FLG'] else num_null_values == 0, f"{col}: {num_null_values} <NA> values"
    return df

def clean_datetime(df, cols):
    date_formats = {
        'STOP_FRISK_DATE': '%Y-%M-%d',  # YYYY-MM-DD
    }
    for col in cols:
        df[col] = pd.to_datetime(df[col], format=date_formats.get(col), errors='coerce')
    return df

def clean_hours(df, cols):
    for col in cols:
        df[col] = df[col].str.split(':').str[0]  # Parse out hours from HH:MM:SS format...
        df[col] = pd.to_numeric(df[col],  errors='coerce')
    return df

def clean_height(df, cols, null_strings=null_strings):
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            # .replace(null_strings, np.nan)
            .str.extract(r'^(\d+)\.?(\d+)?$')  # Extract feet and inches
            .apply(lambda x: (int(x[0]) * 30.48) + (int(x[1]) * 2.54) if pd.notna(x[0]) else np.nan, axis=1)
            .round()
            .astype('Int64')
        )
    return df    
def filter_bad_data(df):
    # Completely arbitrary cut off for bad data, but I'll assume any ages less than 6 to be errors
    # Similarly, this may remove real, valid data for stops of developmentally disabled persons
    df = df.dropna(subset=['SUSPECT_REPORTED_AGE', 'SUSPECT_HEIGHT', 'SUSPECT_RACE_DESCRIPTION'])
    df = df[df['SUSPECT_REPORTED_AGE'] >= 6] 
    df = df[df['SUSPECT_HEIGHT'].between(90, 250)] # Min heights of 3ftm to a max of ~8ft
    return df

def clean_data(df, categories, columns_to_keep=columns_to_keep):
    new_df = df[columns_to_keep].copy()
    new_df = clean_datetime(new_df, categories['datetime'])
    new_df = clean_hours(new_df, categories['hours'])
    new_df = clean_string(new_df, categories['string'])
    new_df = clean_numeric(new_df, categories['numeric'])
    new_df = clean_boolean(new_df, categories['boolean'])
    new_df = clean_height(new_df, categories['height'])
    new_df = filter_bad_data(new_df)
    return new_df

import pandas as pd
from datetime import datetime

def standardize_dates(date_str, output_format='%Y-%m-%d'):
    """
    Convert mixed-format date strings to a consistent format.
    
    Parameters:
        date_str (str): Input date string in various formats
        output_format (str): Desired output format (default: ISO format YYYY-MM-DD)
    
    Returns:
        str: Date in standardized format, or pd.NA if parsing fails
    """
    # Common date formats to try (add more if needed)
    possible_formats = [
        '%m/%d/%Y', '%m/%d/%y',  # US-style dates (10/31/2023 or 10/31/23)
        '%Y-%m-%d', '%y-%m-%d',  # ISO-style dates (2023-10-31 or 23-10-31)
        '%d-%m-%Y', '%d-%m-%y',  # European-style dates (31-10-2023)
        '%b %d, %Y', '%B %d, %Y'  # Text dates (Oct 31, 2023 or October 31, 2023)
    ]
    
    if pd.isna(date_str):
        return pd.NA
    
    for fmt in possible_formats:
        try:
            dt = datetime.strptime(str(date_str), fmt)
            return dt.strftime(output_format)
        except ValueError:
            continue
    
    # If no format worked, return NA (or the original string if you prefer)
    return pd.NA

def is_valid_child(row):
    age = row['SUSPECT_REPORTED_AGE']
    height = row['SUSPECT_HEIGHT']
    
    if age >= 18:
        return True
    # Approximate CDC growth chart percentiles (translates roughly to heights of 92cm-120cm for 6 year olds, 139-210cm)
    expected_min = 2.5*age + 77  # ~1st percentile
    expected_max = 6.5*age + 100  # ~99th percentile
    
    return (height >= expected_min) & (height <= expected_max)


def stateplane_to_latlon(x, y):
    """Convert NY State Plane (ft) coordinates to WGS84 lat/lon"""
    transformer = Transformer.from_crs(2263, 4326, always_xy=True)  # NAD83(ft) → WGS84
    # Note: transform takes (x,y) = (easting,northing) and returns (lon,lat)
    lon, lat = transformer.transform(x, y)
    return pd.Series({'lat': lat, 'lon': lon})

def geocode_df(df):
    # Only process rows with valid coordinates
    coord_mask = (df['STOP_LOCATION_X'].notna() & 
                df['STOP_LOCATION_Y'].notna())
    coord_df = df[coord_mask].copy()

    # Convert coordinates (this may take a minute for large datasets)
    coord_df[['lat', 'lon']] = coord_df.apply(
        lambda row: stateplane_to_latlon(row['STOP_LOCATION_X'], row['STOP_LOCATION_Y']),
        axis=1
    )
    # 1. Check CRS of loaded NTA file
    nta = gpd.read_file('./data/addendum/2020 Neighborhood Tabulation Areas (NTAs)_20250514.geojson')
    print(f"NTA CRS: {nta.crs}")

    # 2. Ensure NTA is in EPSG:4326
    if nta.crs != "EPSG:4326":
        nta = nta.to_crs("EPSG:4326")

    # 3. Check for and fix any invalid geometries
    invalid_geometries = ~nta.is_valid
    if invalid_geometries.any():
        print(f"Found {invalid_geometries.sum()} invalid geometries in NTA data")
        nta['geometry'] = nta.geometry.buffer(0)  # Quick fix for invalid geometries

    # 4. Use 'intersects' predicate instead of 'within'
    geometry = [Point(xy) for xy in zip(coord_df['lon'], coord_df['lat'])]
    stops_gdf = gpd.GeoDataFrame(coord_df, geometry=geometry, crs="EPSG:4326")

    # Spatial join with diagnostic information
    print(f"Stops GDF CRS: {stops_gdf.crs}")
    print(f"NTA CRS: {nta.crs}")

    # Check a sample of point coordinates to verify they look reasonable
    print("Sample of transformed coordinates:")
    print(stops_gdf[['lat', 'lon']].head())

    # Check for points that might be outside NYC bounds
    outside_nyc = stops_gdf[(stops_gdf['lat'] < 40.5) | (stops_gdf['lat'] > 41.0) | 
                            (stops_gdf['lon'] < -74.3) | (stops_gdf['lon'] > -73.7)]
    print(f"Points outside expected NYC bounds: {len(outside_nyc)}")

    # Try the spatial join again with the fixed geometry
    geocoded_df = gpd.sjoin(stops_gdf, nta, how='left', predicate='intersects')
    unmatched = geocoded_df[geocoded_df.index_right.isna()]
    print(f"Unmatched points: {len(unmatched)} out of {len(stops_gdf)} ({len(unmatched)/len(stops_gdf)*100:.2f}%)")
    
    geocoded_df = geocoded_df.drop(unmatched.index)
    unmatched = geocoded_df[geocoded_df.index_right.isna()]
    print(f"Unmatched points remaining after cleaning: {len(unmatched)} out of {len(geocoded_df)} ({len(unmatched)/len(geocoded_df)*100:.2f}%)")

    return geocoded_df

# Official NYC neighborhood-to-borough mapping (as of 2024)
# Source: NYC Planning Community District Profiles
# This mapping is pretty brittle and is sensitive to ordering given the fuzzy matching from nta -> this dictionary
NEIGHBORHOOD_BOROUGH_VALIDATION = {
    # Manhattan
    "Battery Park City": "Manhattan",
    "Chelsea": "Manhattan",
    "Chinatown": "Manhattan",
    "East Harlem": "Manhattan",
    "Financial District": "Manhattan",
    "Greenwich Village": "Manhattan",
    "Harlem": "Manhattan",
    "SoHo": "Manhattan",
    "Upper East Side": "Manhattan",
    "Upper West Side": "Manhattan",
    "Central Harlem": "Manhattan",
    "West Harlem": "Manhattan",
    "Hamilton Heights": "Manhattan",
    "Manhattanville": "Manhattan",
    "Morningside Heights": "Manhattan",
    "Washington Heights": "Manhattan",
    "Inwood": "Manhattan",
    "Hell's Kitchen": "Manhattan",
    "Midtown": "Manhattan",
    "East Village": "Manhattan",
    "Gramercy": "Manhattan",
    "Kips Bay": "Manhattan",
    "Lower East Side": "Manhattan",
    "Murray Hill-Broadway Flushing": "Queens", 
    "Murray Hill": "Manhattan",
    "Broadway Flushing": "Queens",
    "NoHo": "Manhattan",
    "Nolita": "Manhattan",
    "Stuyvesant Town": "Manhattan",
    "Tribeca": "Manhattan",
    "West Village": "Manhattan",
    "Yorkville": "Manhattan",
    "Lenox Hill": "Manhattan",
    "Lincoln Square": "Manhattan",
    "Sutton Place": "Manhattan",
    "Carnegie Hill": "Manhattan",
    "Turtle Bay": "Manhattan",
    "Tudor City": "Manhattan",
    "Flatiron District": "Manhattan",
    "Hudson Yards": "Manhattan",
    "NoMad": "Manhattan",
    "Roosevelt Island": "Manhattan",
    "Two Bridges": "Manhattan",
   
    # Brooklyn
    "Bedford-Stuyvesant": "Brooklyn",
    "Bensonhurst": "Brooklyn",
    "Brooklyn Heights": "Brooklyn",
    "Bushwick": "Brooklyn",
    "Coney Island": "Brooklyn",
    "DUMBO": "Brooklyn",
    "East New York": "Brooklyn",
    "Flatbush": "Brooklyn",
    "Park Slope": "Brooklyn",
    "Williamsburg": "Brooklyn",
    "Bay Ridge": "Brooklyn",
    "Boerum Hill": "Brooklyn",
    "Borough Park": "Brooklyn",
    "Brighton Beach": "Brooklyn",
    "Brownsville": "Brooklyn",
    "Canarsie": "Brooklyn",
    "Carroll Gardens": "Brooklyn",
    "Clinton Hill": "Brooklyn",
    "Cobble Hill": "Brooklyn",
    "Crown Heights": "Brooklyn",
    "Highland Park-Cypress Hills Cemeteries (North)": "Queens",
    "Cypress Hills": "Brooklyn",
    "Highland Park": "Brooklyn",
    "Cypress Hills Cemeteries": "Queens",
    "Dyker Heights": "Brooklyn",
    "East Flatbush": "Brooklyn",
    "Fort Greene": "Brooklyn",
    "Gowanus": "Brooklyn",
    "Gravesend": "Brooklyn",
    "Greenpoint": "Brooklyn",
    "Kensington": "Brooklyn",
    "Midwood": "Brooklyn",
    "Mill Basin": "Brooklyn",
    "Ocean Hill": "Brooklyn",
    "Prospect Heights": "Brooklyn",
    "Prospect Lefferts Gardens": "Brooklyn",
    "Red Hook": "Brooklyn",
    "Sheepshead Bay": "Brooklyn",
    "Sunset Park": "Brooklyn",
    "Windsor Terrace": "Brooklyn",
    "Flatlands": "Brooklyn",
    "Marine Park": "Brooklyn",
    "Bergen Beach": "Brooklyn",
    "Sea Gate": "Brooklyn",
    "Manhattan Beach": "Brooklyn",
    "Gerritsen Beach": "Brooklyn",
    "Georgetown": "Brooklyn",
    "Starrett City": "Brooklyn",
    "Bath Beach": "Brooklyn",
   
    # Queens
    "Astoria": "Queens",
    "Flushing": "Queens",
    "Jackson Heights": "Queens",
    "Long Island City": "Queens",
    "Sunnyside": "Queens",
    "Bayside": "Queens",
    "Corona": "Queens",
    "East Elmhurst": "Queens",
    "Elmhurst": "Queens",
    "Forest Hills": "Queens",
    "Fresh Meadows": "Queens",
    "Glendale": "Queens",
    "Howard Beach": "Queens",
    "Jamaica": "Queens",
    "Kew Gardens": "Queens",
    "Little Neck": "Queens",
    "Maspeth": "Queens",
    "Middle Village": "Queens",
    "Ozone Park": "Queens",
    "Queens Village": "Queens",
    "Rego Park": "Queens",
    "Richmond Hill": "Queens",
    "Ridgewood": "Queens",
    "Rockaway Beach": "Queens",
    "Whitestone": "Queens",
    "Woodhaven": "Queens",
    "Woodside": "Queens",
    "Auburndale": "Queens",
    "Bayswater": "Queens",
    "Bellerose": "Queens",
    "Briarwood": "Queens",
    "Cambria Heights": "Queens",
    "College Point": "Queens",
    "Douglaston": "Queens",
    "Far Rockaway": "Queens",
    "Floral Park": "Queens",
    "Glen Oaks": "Queens",
    "Hollis": "Queens",
    "Hollis Hills": "Queens",
    "Jamaica Estates": "Queens",
    "Jamaica Hills": "Queens",
    "Kew Gardens Hills": "Queens",
    "Laurelton": "Queens",
    "Rosedale": "Queens",
    "South Jamaica": "Queens",
    "Springfield Gardens": "Queens",
    "St. Albans": "Queens",
    "Utopia": "Queens",
    "Queensbridge": "Queens",
    "Ravenswood": "Queens",
    "Dutch Kills": "Queens",
    "Queensbridge-Ravenswood-Dutch Kills": "Queens",
   
    # Bronx
    "Fordham": "Bronx",
    "Hunts Point": "Bronx",
    "Mott Haven": "Bronx",
    "Pelham Bay": "Bronx",
    "University Heights": "Bronx",
    "Bedford Park": "Bronx",
    "Belmont": "Bronx",
    "Castle Hill": "Bronx",
    "City Island": "Bronx",
    "Claremont Village": "Bronx",
    "Co-op City": "Bronx",
    "Concourse": "Bronx",
    "Concourse Village": "Bronx",
    "Eastchester": "Bronx",
    "Edenwald": "Bronx",
    "Fieldston": "Bronx",
    "Highbridge Park": "Manhattan",
    "Highbridge": "Bronx",
    "Kingsbridge": "Bronx",
    "Kingsbridge Heights": "Bronx",
    "Melrose": "Bronx",
    "Morris Heights": "Bronx",
    "Morris Park": "Bronx",
    "Morrisania": "Bronx",
    "Norwood": "Bronx",
    "Parkchester": "Bronx",
    "Pelham Gardens": "Bronx",
    "Pelham Parkway": "Bronx",
    "Port Morris": "Bronx",
    "Riverdale": "Bronx",
    "Soundview": "Bronx",
    "Spuyten Duyvil": "Bronx",
    "Throgs Neck": "Bronx",
    "Tremont": "Bronx",
    "Van Nest": "Bronx",
    "Wakefield": "Bronx",
    "West Farms": "Bronx",
    "Westchester Square": "Bronx",
    "Williamsbridge": "Bronx",
    "Woodlawn": "Bronx",
    "Baychester": "Bronx",
    "Allerton": "Bronx",
    "Clason Point": "Bronx",
    "Olinville": "Bronx",
    "Mount Hope": "Bronx",
    "Mount Eden-Claremont (West)": "Bronx",
    "Mount Eden": "Bronx",
    "Claremont": "Bronx",
    "Claremont Village": "Bronx",
    "Crotona Park East": "Bronx",
    "Longwood": "Bronx",
   
    # Staten Island
    "Port Richmond": "Staten Island",
    "St. George": "Staten Island",
    "Tottenville": "Staten Island",
    "Annadale": "Staten Island",
    "Arden Heights": "Staten Island",
    "Arlington": "Staten Island",
    "Arrochar": "Staten Island",
    "Bay Terrace (Staten Island)": "Staten Island",
    "Bulls Head": "Staten Island",
    "Castleton Corners": "Staten Island",
    "Charleston": "Staten Island",
    "Clifton": "Staten Island",
    "Dongan Hills": "Staten Island",
    "Eltingville": "Staten Island",
    "Emerson Hill": "Staten Island",
    "Fort Wadsworth": "Staten Island",
    "Graniteville": "Staten Island",
    "Grant City": "Staten Island",
    "Grasmere": "Staten Island",
    "Great Kills": "Staten Island",
    "Grymes Hill": "Staten Island",
    "Huguenot": "Staten Island",
    "Lighthouse Hill": "Staten Island",
    "Livingston": "Staten Island",
    "Mariners Harbor": "Staten Island",
    "Midland Beach": "Staten Island",
    "New Brighton": "Staten Island",
    "New Dorp": "Staten Island",
    "New Springville": "Staten Island",
    "Oakwood": "Staten Island",
    "Ocean Breeze": "Staten Island",
    "Old Town": "Staten Island",
    "Pleasant Plains": "Staten Island",
    "Prince's Bay": "Staten Island",
    "Randall Manor": "Staten Island",
    "Richmond Valley": "Staten Island",
    "Richmondtown": "Staten Island",
    "Rosebank": "Staten Island",
    "Rossville": "Staten Island",
    "Sandy Ground": "Staten Island",
    "Shore Acres": "Staten Island",
    "Silver Lake": "Staten Island",
    "South Beach": "Staten Island",
    "Stapleton": "Staten Island",
    "Todt Hill": "Staten Island",
    "Tompkinsville": "Staten Island",
    "Travis": "Staten Island",
    "Ward Hill": "Staten Island",
    "West New Brighton": "Staten Island",
    "Westerleigh": "Staten Island",
    "Willowbrook": "Staten Island",
    "Woodrow": "Staten Island",
    "Freshkills Park (North)": "Staten Island",
    "Freshkills Park (South)": "Staten Island",
    "Miller Field": "Staten Island",
    "Fort Hamilton": "Brooklyn",
    "Brooklyn Navy Yard": "Brooklyn",
    "Green-Wood Cemetery": "Brooklyn",
    "Barren Island-Floyd Bennett Field": "Brooklyn",
    "Prospect Park": "Brooklyn",
    "Lincoln Terrace Park": "Brooklyn",
    "Madison": "Brooklyn",
    "Bay Terrace-Clearview": "Queens",
    "St. John Cemetery": "Queens",
    "Baisley Park": "Queens",
    "Pomonok-Electchester-Hillcrest": "Queens",
    "Pomonok": "Queens",
    "Electchester": "Queens",
    "Hillcrest": "Queens",
    "Kissena Park": "Queens",
    "Forest Park": "Queens",
    "Cunningham Park": "Queens",
    "Alley Pond Park": "Queens",
    "Queensboro Hill": "Queens",
    "LaGuardia Airport": "Queens",
    "John F. Kennedy International Airport": "Queens",
    "Jacob Riis Park-Fort Tilden-Breezy Point Tip": "Queens",
    "Breezy Point-Belle Harbor-Rockaway Park-Broad Channel": "Queens",
    "Breezy Point": "Queens",
    "Belle Harbor": "Queens",
    "Rockaway Park": "Queens",
    "Broad Channel": "Queens",
    "Bronx Park": "Bronx",
    "Crotona Park": "Bronx",
    "Van Cortlandt Park": "Bronx",
    "Yankee Stadium-Macombs Dam Park": "Bronx",
    "Hutchinson Metro Center": "Bronx",
    "Central Park": "Manhattan",
    "The Battery-Governors Island-Ellis Island-Liberty Island": "Manhattan",
    "The Battery": "Manhattan",
    "Governors Island": "Manhattan",
    "Ellis Island": "Manhattan",
    "Liberty Island": "Manhattan", 
    "Randall's Island": "Manhattan"
}

def test_neighborhood_borough_mapping(joined_data):
    """Validate that all neighborhoods map to correct boroughs"""
    errors = []
    not_matched = []
    
    for _, row in joined_data.iterrows():
        nta_name = row['ntaname']
        mapped_borough = row['boroname']
        
        if pd.isna(nta_name):
            continue  # Skip null neighborhoods
        
        # Find matching neighborhood in validation dict (case-insensitive)
        matched = False
        for valid_hood, valid_borough in NEIGHBORHOOD_BOROUGH_VALIDATION.items():
            if valid_hood.lower() in nta_name.lower():
                expected_borough = valid_borough.title()  # Standardize casing
                if mapped_borough.lower() != expected_borough.lower():
                    errors.append({
                        'stop_id': row['STOP_ID'],
                        'nta_name': nta_name,
                        'mapped_borough': mapped_borough,
                        'expected_borough': expected_borough
                    })
                matched = True
                break
        
        if not matched:
            print(f"Warning: No validation rule for {nta_name}")
            not_matched.append(nta_name)
    
    return errors, not_matched

# Run the test on your joined data
def validate_neighborhood_data(df):
    validation_errors, unmatched_ntas = test_neighborhood_borough_mapping(df)

    if validation_errors:
        print(f"Found {len(validation_errors)} borough mapping errors:")
        error_df = pd.DataFrame(validation_errors)
        print(error_df.head(len(validation_errors)))
        
        # Calculate error rate
        total_mapped = len(df[df['ntaname'].notna()])
        error_rate = len(validation_errors) / total_mapped
        print(f"\nError rate: {error_rate:.2%}")
    else:
        print("All neighborhood-borough mappings validated successfully!")

    print(f"Unique missing NTAs:", set(unmatched_ntas))
    
    mismatches = df[
        df['STOP_LOCATION_BORO_NAME'].str.lower() != 
        df['boroname'].str.lower()
    ]

    # Display the mismatches
    print(f"Found {len(mismatches)} borough name mismatches:")
    print(mismatches[['STOP_ID', 'ntaname', 'STOP_LOCATION_BORO_NAME', 'boroname']].head())

    # While I'm tempted to override these errors, I want to keep the data as unmodified as possible, so I'm dropping these
    df_without_mismatches = df.drop(labels=mismatches.index)

    mismatches = df_without_mismatches[
        df_without_mismatches['STOP_LOCATION_BORO_NAME'].str.lower() != 
        df_without_mismatches['boroname'].str.lower()
    ]

    # Display the mismatches
    print(f"After cleaning: Found {len(mismatches)} borough name mismatches:")
    print(mismatches[['STOP_ID', 'ntaname', 'STOP_LOCATION_BORO_NAME', 'boroname']].head())

def prepare_additional_features(df):
    final_df = df.copy()
    final_df = final_df.rename(columns={"ntaname" : "NEIGHBORHOOD"})

    final_df['OUTCOME_OF_STOP'] = np.select(
        condlist=[
            final_df['SUSPECT_ARRESTED_FLAG'],  
            final_df['SUMMONS_ISSUED_FLAG']
        ],
        choicelist=['Arrested', 'Summoned'],
        default='No Charges Filed'
    )

    PHYSICAL_FORCE_COLUMNS = [
        'PHYSICAL_FORCE_CEW_FLAG',            # Conducted Energy Weapon (Taser)
        'PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG',
        'PHYSICAL_FORCE_HANDCUFF_SUSPECT_FLAG',
        'PHYSICAL_FORCE_OC_SPRAY_USED_FLAG',  # Pepper spray
        'PHYSICAL_FORCE_OTHER_FLAG',
        'PHYSICAL_FORCE_RESTRAINT_USED_FLAG',
        # 'PHYSICAL_FORCE_VERBAL_INSTRUCTION_FLAG', # Omitting as this doesn't reflect the use of force, but a warning.  Also distorts the dataset...
        'PHYSICAL_FORCE_WEAPON_IMPACT_FLAG'
    ]

    # Create aggregated force column (True if ANY force was used)
    final_df['OFFICER_USED_FORCE'] = (
        final_df[PHYSICAL_FORCE_COLUMNS]
        .fillna(False)                # Treat missing values as no force
        .any(axis=1)                  # True if any force column is True
        .astype(bool)                 # Ensure boolean dtype
    )

    final_df['FORCE_TYPE'] = np.select(
        [
            final_df['PHYSICAL_FORCE_WEAPON_IMPACT_FLAG'],
            final_df['PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG'],
            final_df['PHYSICAL_FORCE_CEW_FLAG'], 
            final_df['PHYSICAL_FORCE_OC_SPRAY_USED_FLAG'],
            final_df['PHYSICAL_FORCE_HANDCUFF_SUSPECT_FLAG'],
            final_df['PHYSICAL_FORCE_OTHER_FLAG'],
            # final_df['PHYSICAL_FORCE_VERBAL_INSTRUCTION_FLAG'],
            final_df['PHYSICAL_FORCE_RESTRAINT_USED_FLAG']
        ],
        [
            'Weapon Impact',
            'Firearm Drawn', 
            'Taser',
            'Pepper Spray',
            'Handcuffs',
            'Other Physical Force',
            # 'Verbal Commands', 
            'Restraint Used'
        ],
        default='No Force'
    )

    # Verification
    final_df = final_df.reset_index(drop=True)

    assert len(final_df['SUSPECT_ARRESTED_FLAG']) == len(final_df)
    assert len(final_df['SUMMONS_ISSUED_FLAG']) == len(final_df)
    assert final_df[['OUTCOME_OF_STOP']].notna().all().all(), "Null values detected"
    print(final_df[['OUTCOME_OF_STOP']].value_counts())
    print(final_df[['OFFICER_USED_FORCE']].value_counts())
    print(final_df[['FORCE_TYPE']].value_counts())
    return final_df