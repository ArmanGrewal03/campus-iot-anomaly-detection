import csv
import random

buildings = [
    "AOB Atrium on Bay, 20 Dundas Street West",
    "ARC Architecture Building — Paul H. Cocker Gallery, 325 Church Street",
    "BKS Campus Store, 17 Gould Street",
    "BND 114 Bond Street",
    "BON 111 Bond Street",
    "BTS Bell Trinity Square, 483 Bay Street",
    "CAR Carlton Cinema, 20 Carlton Street",
    "CED The Chang School of Continuing Education (Heaslip House), 297 Victoria Street",
    "CIS Creative Innovation Studio, 110 Bond Street",
    "CIV Civil Engineering Storage, 106 Mutual Street",
    "COP 101 Gerrard Street East",
    "CPK English Language Institute and International College (College Park), 424 Yonge Street",
    "CUI Centre for Urban Innovation, 44 Gerrard Street East",
    "DAL 147 Dalhousie Street",
    "DCC Daphne Cockwell Health Sciences Complex, 288 Church Street",
    "DSQ Yonge-Dundas Square, 10 Dundas Street East",
    "ENG George Vari Engineering and Computing Centre, 245 Church Street",
    "EPH Eric Palin Hall, 87 Gerrard Street East",
    "HEI School of Graphic Communications Management (Heidelberg Centre), 125 Bond Street",
    "ILC International Living / Learning Centre, 133 Mutual Street and 240 Jarvis Street",
    "IMA School of Image Arts, 122 Bond Street",
    "IMC The Image Centre, 33 Gould Street",
    "JOR Jorgenson Hall, 380 Victoria Street",
    "KHE Kerr Hall East, 340 Church Street",
    "KHN Kerr Hall North, 31 / 43 Gerrard Street East",
    "KHS Kerr Hall South, 40 / 50 / 60 Gould Street",
    "KHW Kerr Hall West, 379 Victoria Street",
    "LIB Library Building, 350 Victoria Street",
    "MAC Mattamy Athletic Centre, 50 Carlton Street",
    "MER Merchandise Building, 159 Dalhousie Street",
    "MON Civil Engineering Building (Monetary Times), 341 Church Street",
    "MRS MaRS Building, 661 University Avenue",
    "OAK Oakham House, 63 Gould Street",
    "OKF O’Keefe House, 137 Bond Street",
    "PIT Pitman Hall, 160 Mutual Street",
    "PKG Parking Garage, 300 Victoria Street",
    "POD Podium, 350 Victoria Street",
    "PRO 112 Bond Street",
    "RAC Recreation and Athletics Centre, 40 / 50 Gould Street",
    "RCC Rogers Communications Centre, 80 Gould Street",
    "SBB South Bond Building, 105 Bond Street",
    "SCC Student Campus Centre, 55 Gould Street",
    "SHE Sally Horsfall Eaton Centre for Studies in Community Health, 99 Gerrard Street East",
    "SID School of Interior Design, 302 Church Street",
    "SLC Sheldon & Tracy Levy Student Learning Centre, 341 Yonge Street",
    "SMH St. Michael’s Hospital, 209 Victoria Street",
    "TEC Toronto Eaton Centre, 220 Yonge Street",
    "TRS Ted Rogers School of Management, 55 Dundas Street West",
    "VIC Victoria Building, 285 Victoria Street",
    "YDI Yonge-Dundas Intersection, 1 Dundas St West",
    "YNG 415 Yonge Street"
]

input_file = "UNSW-NB15_4.csv"
output_file = "UNSW-NB15_Location.csv"

ip_to_location = {}

with open(input_file, 'r', newline='') as infile:
    reader = csv.reader(infile)
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            if i >= 400:
                break
            
            srcip = row[0]
            if srcip not in ip_to_location:
                ip_to_location[srcip] = random.choice(buildings)
                
            new_row = row + [ip_to_location[srcip]]
            writer.writerow(new_row)

print(f"Successfully generated {output_file} with {min(400, i)} lines.")
