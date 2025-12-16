import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="API Demo", page_icon="🔗", layout="wide")

st.title("🔗 API Integration Examples")
st.markdown("Working with public APIs to fetch and visualize real-time data")

# Sidebar
st.sidebar.header("API Selection")
api_choice = st.sidebar.radio(
    "Choose an API:",
    ["Public Holidays", "Random User", "Cat Facts", "IP Information"]
)

st.markdown("---")

if api_choice == "Public Holidays":
    st.subheader("🎉 Public Holidays API")
    st.markdown("Fetch public holidays for any country using the [Abstract API](https://www.abstractapi.com/)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        year = st.selectbox("Select Year", [2024, 2025, 2026])
        country = st.selectbox("Select Country", 
            ["US", "GB", "CA", "AU", "DE", "FR", "JP", "IN"],
            format_func=lambda x: {
                "US": "🇺🇸 United States",
                "GB": "🇬🇧 United Kingdom", 
                "CA": "🇨🇦 Canada",
                "AU": "🇦🇺 Australia",
                "DE": "🇩🇪 Germany",
                "FR": "🇫🇷 France",
                "JP": "🇯🇵 Japan",
                "IN": "🇮🇳 India"
            }[x]
        )
        
        fetch_button = st.button("📥 Fetch Holidays", type="primary")
    
    with col2:
        if fetch_button:
            with st.spinner("Fetching holidays..."):
                try:
                    # Using Nager.Date API (free, no key needed)
                    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        holidays = response.json()
                        
                        st.success(f"✅ Found {len(holidays)} holidays for {country} in {year}")
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(holidays)
                        df['date'] = pd.to_datetime(df['date'])
                        df['month'] = df['date'].dt.month_name()
                        
                        # Display table
                        display_df = df[['date', 'localName', 'name']].copy()
                        display_df.columns = ['Date', 'Local Name', 'English Name']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Visualization
                        st.markdown("#### Holidays by Month")
                        monthly_counts = df.groupby('month').size().reset_index(name='count')
                        
                        fig = px.bar(
                            monthly_counts,
                            x='month',
                            y='count',
                            color='count',
                            color_continuous_scale='Viridis',
                            title=f'Distribution of Holidays in {year}'
                        )
                        fig.update_layout(
                            xaxis_title='Month',
                            yaxis_title='Number of Holidays',
                            showlegend=False,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Failed to fetch data. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif api_choice == "Random User":
    st.subheader("👤 Random User Generator")
    st.markdown("Generate random user profiles using [RandomUser.me API](https://randomuser.me/)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_users = st.slider("Number of users", 1, 10, 5)
        nationality = st.selectbox("Nationality", 
            ["Random", "US", "GB", "CA", "AU", "FR", "DE", "BR"],
            format_func=lambda x: "🌍 " + x if x == "Random" else x
        )
        
        generate_button = st.button("🎲 Generate Users", type="primary")
    
    with col2:
        if generate_button:
            with st.spinner("Generating users..."):
                try:
                    nat_param = "" if nationality == "Random" else f"&nat={nationality}"
                    url = f"https://randomuser.me/api/?results={num_users}{nat_param}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        users = data['results']
                        
                        st.success(f"✅ Generated {len(users)} random users")
                        
                        # Display users
                        for i, user in enumerate(users, 1):
                            with st.expander(f"👤 {user['name']['first']} {user['name']['last']}", expanded=i==1):
                                user_col1, user_col2 = st.columns([1, 2])
                                
                                with user_col1:
                                    st.image(user['picture']['large'], width=150)
                                
                                with user_col2:
                                    st.markdown(f"**Email:** {user['email']}")
                                    st.markdown(f"**Phone:** {user['phone']}")
                                    st.markdown(f"**Location:** {user['location']['city']}, {user['location']['country']}")
                                    st.markdown(f"**Age:** {user['dob']['age']}")
                                    st.markdown(f"**Username:** {user['login']['username']}")
                    else:
                        st.error("❌ Failed to generate users. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif api_choice == "Cat Facts":
    st.subheader("🐱 Random Cat Facts")
    st.markdown("Get interesting facts about cats from [Cat Facts API](https://catfact.ninja/)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_facts = st.slider("Number of facts", 1, 10, 3)
        fetch_button = st.button("🐾 Get Cat Facts", type="primary")
    
    with col2:
        if fetch_button:
            with st.spinner("Fetching cat facts..."):
                try:
                    facts = []
                    for _ in range(num_facts):
                        response = requests.get("https://catfact.ninja/fact", timeout=10)
                        if response.status_code == 200:
                            facts.append(response.json()['fact'])
                    
                    if facts:
                        st.success(f"✅ Retrieved {len(facts)} cat facts!")
                        
                        for i, fact in enumerate(facts, 1):
                            st.info(f"**Fact #{i}:** {fact}")
                    else:
                        st.error("❌ Failed to fetch cat facts.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif api_choice == "IP Information":
    st.subheader("🌐 IP Address Information")
    st.markdown("Get geolocation data for any IP address using [ipapi.co](https://ipapi.co/)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Options")
        use_own_ip = st.checkbox("Use my IP address", value=True)
        
        if not use_own_ip:
            ip_address = st.text_input("Enter IP Address", placeholder="8.8.8.8")
        else:
            ip_address = None
        
        lookup_button = st.button("🔍 Lookup", type="primary")
    
    with col2:
        if lookup_button:
            with st.spinner("Looking up IP information..."):
                try:
                    if ip_address:
                        url = f"https://ipapi.co/{ip_address}/json/"
                    else:
                        url = "https://ipapi.co/json/"
                    
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'error' in data:
                            st.error(f"❌ {data['reason']}")
                        else:
                            st.success("✅ IP information retrieved!")
                            
                            # Display info
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.markdown(f"**IP Address:** {data.get('ip', 'N/A')}")
                                st.markdown(f"**City:** {data.get('city', 'N/A')}")
                                st.markdown(f"**Region:** {data.get('region', 'N/A')}")
                                st.markdown(f"**Country:** {data.get('country_name', 'N/A')} {data.get('country_emoji', '')}")
                            
                            with info_col2:
                                st.markdown(f"**Postal Code:** {data.get('postal', 'N/A')}")
                                st.markdown(f"**Timezone:** {data.get('timezone', 'N/A')}")
                                st.markdown(f"**Currency:** {data.get('currency', 'N/A')}")
                                st.markdown(f"**ISP:** {data.get('org', 'N/A')}")
                            
                            # Map
                            if data.get('latitude') and data.get('longitude'):
                                st.markdown("#### Location on Map")
                                map_df = pd.DataFrame({
                                    'lat': [data['latitude']],
                                    'lon': [data['longitude']]
                                })
                                st.map(map_df, zoom=10)
                    else:
                        st.error("❌ Failed to lookup IP information.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Information
st.markdown("---")
st.info("""
💡 **About these APIs:**
- All APIs used are free and don't require authentication
- Real-time data is fetched when you click the action buttons
- These examples demonstrate how to integrate external data sources into applications
- Error handling is implemented for robust API interactions
""")