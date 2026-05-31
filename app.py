import streamlit as st
import pandas as pd
import sklearn
import joblib
from xgboost import XGBClassifier 

st.set_page_config(page_title="FloodAle",layout='centered')

#Page Header
st.header("FloodAle | Flood Prediction System",divider=True)
st.write("Enter data and get flood prediciton")

#All Data inputs
city = st.selectbox(
    "Select your city",
    options=['Barisal', 'Bhola', 'Bogra', 'Chandpur',
       'Chittagong (City-Ambagan)', 'Chittagong (IAP-Patenga)', 'Comilla',
       "Cox's Bazar", 'Dhaka', 'Dinajpur', 'Faridpur', 'Feni', 'Hatiya',
       'Ishurdi', 'Jessore', 'Khepupara', 'Khulna', 'Kutubdia',
       'Madaripur', 'Maijdee Court', 'Mongla', 'Mymensingh', 'Patuakhali',
       'Rajshahi', 'Rangamati', 'Rangpur', 'Sandwip', 'Satkhira',
       'Sitakunda', 'Srimangal', 'Sylhet', 'Tangail', 'Teknaf'],
)

year = (st.text_input("Year"))
month = st.selectbox(
    "Select month",
    options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
)
max_temp = (st.number_input("Max Temp (Celcius)"))
min_temp = (st.number_input("Min Temp (Celcius)"))
rainfall = (st.number_input("Rainfall (cm)"))
humidity = (st.number_input("Relative Humidity (Percentage)"))
wind = (st.number_input("Wind Speed (m/s)"))
cloud = (st.number_input("Cloud Converge (Octs)"))
sunshine = (st.number_input("Bright Sunshine (hours)"))

#X,Y,Latitude,Longitude,Alt of 32 stations
city_info = {
    "Bogra" : [435303.70,	751187.5,	24.88,	89.3600, 20],
    "Satkhira" : [404211.70, 508367.2, 	22.68,	89.0700, 6],
    "Comilla":[621445.20	, 596963.4,	23.48,	91.1900	,10],
    "Cox's Bazar":[705183.00,	374324.6,	21.46,	91.9800,	4],
    "Dinajpur":[365790.20,	834718.0,	25.63,	88.6600,	37],
    "Jessore":[	420062.70,	562498.8,	23.17,	89.2200,	7],
    "Khulna":[456632.40, 521635.7,	22.80,	89.5800,	4],
    "Faridpur":[ 483877.30,	610719.3,	23.61,	89.8400,	9],
    "Mymensingh":[	540975.30,	737535.8,	24.75,	90.4100,	19],
    "Srimangal":[	675761.60,	687095.9,	24.29,	91.7300,	23],
    "Barisal":[	536809.80,	510151.9,	22.70,	90.3600,	4],
    "Chittagong (IAP-Patenga)":[684570.90,	471415.0,	22.34,	91.7900, 6],
    "Maijdee Court":[610851.80,	524433.6,	22.83,	91.0800,	6],
    "Dhaka":[540098.60,	629248.4,	23.78,	90.3900,	9],
    "Rangpur":[	426172.90,	844822.3,	25.72,	89.2600, 34],
    "Sylhet":[	694533.20,	752277.9,	24.88,	91.9300,	35],
    "Rangamati":[	725903.30,	508021.9,	22.67,	92.2000,	63],
    "Ishurdi":[	402483.20	,667640.8,	24.12,	89.0400,	14],
    "Chandpur":[	568556.90,	571945.8,	23.26,	90.6700,	7],
    "Rajshahi":[	353944.90	,693386.6,	24.35,	88.5600,	20],
    "Sandwip":[	650012.10,	488627.9,	22.50,	91.4600	,6],
    "Bhola":[	567637.60,	510271.8,	22.70,	90.6600,	5],
    "Hatiya":[	616159.20	,465295.1,	22.29,	91.1300,	4],
    "Feni":[	640285.90	,544954.5,	23.01,	91.3700,	8],
    "Patuakhali":[	534986.10,	472575.7,	22.36,	90.3400,	3],
    "Khepupara":[	522893.10,	430006.8,	21.98,	90.2200,	3],
    "Teknaf":[	734765.40	,308914.1,	20.87,	92.2600	,4],
    "Kutubdia":[	690250.30	,414997.4,	21.83,	91.8400,	6],
    "Madaripur":[	518762.70	,561770.3,	23.17,	90.1800,	5],
    "Sitakunda":[	668856.20,	504500.3,	22.64,	91.6400,	4],
    "Tangail":[	491982.91,	683166.0,	24.15,	89.5500,	10],
    "Mongla":[	452007.30	,499110.5,	22.43,	89.6600	,4],
    "Chittagong (City-Ambagan)":[	0.00,	0.0,	22.35,	91.8166,	0],
}

def month_num(month):
    if month=='January':
        return 1
    elif month=='February':
        return 2
    elif month=='March':
        return 3
    elif month=='April':
        return 4
    elif month=='May':
        return 5
    elif month=='June':
        return 6
    elif month=='July':
        return 7
    elif month=='August':
        return 8
    elif month=='September':
        return 9
    elif month=='October':
        return 10
    elif month=='September':
        return 11
    else:
        return 12
    	

def get(city,i):
    if i=='X':
        i=0
    elif i=='Y':
        i=1
    elif i=='L':
        i=2
    elif i=='G':
        i=3
    elif i=='A':
        i=4

    return city_info[city][i]
    
    
def getPer(y,m):
    n = m/100
    y = int(y)
    return y+n

def polish(year,month,max_temp,min_temp,rainfall,humidity,wind,cloud,sunshine):
    return pd.DataFrame(
        {
            'Year': [int(year)],
            'Month': [int(month_num(month))],
            'Max_Temp': [float(max_temp)],
            'Min_Temp': [float(min_temp)],
            'Rainfall': [float(rainfall)],
            'Relative_Humidity': [float(humidity)],
            'Wind_Speed': [float(wind)],
            'Cloud_Coverage': [float(cloud)],
            'Bright_Sunshine': [float(sunshine)],
            'X_COR': [float(get(city,'X'))],
            'Y_COR': [float(get(city,'Y'))],
            'LATITUDE': [float(get(city,'L'))],
            'LONGITUDE': [float(get(city,'G'))],
            'ALT': [int(get(city,'A'))],
            'Period': [float(getPer(year,month_num(month)))]
        }
    )


test_flood = pd.DataFrame(
    {
	   'Year':[2023,2022,2023],
       'Month': [1, 6, 8],
       'Max_Temp': [26.48, 29.85,31.26],
       'Min_Temp': [13.92, 24.54, 26.13],
       'Rainfall': [0, 148.50, 109.80],
       'Relative_Humidity': [72.97, 89.87, 85.47],
       'Wind_Speed': [2.29, 3.65, 3.08],
       'Cloud_Coverage': [0.72, 7.70, 6.94],
       'Bright_Sunshine': [6.51, 1.23, 2.53],
       'X_COR':[694533.2, 694533.2, 684570.9],
       'Y_COR':[752277.9, 752277.9, 471415.0],
       'LATITUDE':[24.88, 24.88, 22.34],
       'LONGITUDE':[91.93, 91.93, 91.79],
       'ALT':[35,35, 6],
       'Period':[2023.01, 2022.06, 2023.08]
    }
)

load = joblib.load('FloodAle.joblib')
#ans = load.predict(test_flood)
#print(ans)

if st.button('Show Prediction'):
    
    if ((city=="") or (year=="") or (month=="") or (max_temp=="") or (min_temp=="") or (rainfall=="") or (humidity=="") or (wind=="") or (cloud=="") or (sunshine=="")):
        st.warning('Enter all data')
    else:
        c = polish(year,month,max_temp,min_temp,rainfall,humidity,wind,cloud,sunshine)
        pred = load.predict(c)
        if pred==1:
            st.info("Flood")
            st.toast("Flood! Be prepared!")
        elif pred==0:
            no = st.success("No flood")
            st.balloons()
            st.toast("No Worries, No Flood!")
        
     

    