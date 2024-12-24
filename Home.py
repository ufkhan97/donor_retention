import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(page_title="Donor Cohort Dashboard", layout="wide")

# Title and description
st.title("Gitcoin Grants Donor Cohort Analysis")
st.markdown("Analysis of donor behavior and retention across Gitcoin Grants rounds. This does not include rounds run by our partners such as Zuzalu, SEI, etc")

# Move filters to sidebar
st.sidebar.header("Filters")
FILTER_TO_FC = st.sidebar.toggle("Filter to Farcaster Users (for retention analysis)", value=False)
FILTER_TO_GS = st.sidebar.toggle("Filter to Recent Rounds (16+)", value=False)
FILTER_TO_PASSPORT = st.sidebar.toggle("Filter to Passport Holders", value=False)

def format_graph(fig, title, x_title, y_title, y2_title=None):
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='DarkSlateGrey'),
            pad=dict(t=20, b=20)
        ),
        xaxis=dict(
            title=x_title,
            titlefont=dict(size=16, color='DarkSlateGrey'),
            tickfont=dict(size=14, color='DarkSlateGrey'),
            showgrid=False,
            showline=True,
            linecolor='DarkSlateGrey'
        ),
        yaxis=dict(
            title=y_title,
            titlefont=dict(size=16, color='DarkSlateGrey'),
            tickfont=dict(size=14, color='DarkSlateGrey'),
            showgrid=False,
            showline=True,
            linecolor='DarkSlateGrey'
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified',
        legend=dict(
            x=0.05, y=1.05,  # Adjust these values to move the legend
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='DarkSlateGrey',
            borderwidth=0,
            orientation="h"  
        )
    )
    if y2_title:
        fig.update_layout(
            yaxis2=dict(
                title=y2_title,
                titlefont=dict(size=16),
                tickfont=dict(size=14),
                overlaying='y',
                side='right',
                showgrid=False,
                showline=True,
                linecolor='DarkSlateGrey'
            )
        )
    return fig

# Load data in a function
@st.cache_data  # Cache the data loading
def load_data():
    gg_voters = pd.read_csv('data/midgg22_donors.csv')
    gg22_voters = pd.read_csv('data/gg22_voters_for_retention_analysis_end.csv')
    return gg_voters, gg22_voters

# Data processing in a function
def process_data(gg_voters, gg22_voters, filter_fc, filter_gs, filter_passport):
    # Combine and filter data
    all_voters = pd.concat([gg_voters, gg22_voters])
    all_voters = all_voters[all_voters['voter'].str.startswith('0x') & (all_voters['voter'].str.len() == 42)]
    
    # Apply filters based on toggles
    if filter_fc:
        fc = pd.read_csv('data/farcaster_may9_oso.csv')
        all_voters = all_voters[all_voters['voter'].isin(fc['address'])]

    if filter_gs:
        all_voters = all_voters[all_voters['round_num'] >= 16]

    if filter_passport:
        passport = pd.read_csv('data/passport_scores_june24.csv')
        passport = passport[passport['rawScore'] >= 1]
        all_voters = all_voters[all_voters['voter'].isin(passport['address'])]
    
    # Process round numbers and create cohorts
    all_voters['round_num'] = pd.to_numeric(all_voters['round_num'], errors='coerce')
    all_voters = all_voters.dropna(subset=['round_num'])
    all_voters['cohort'] = all_voters.groupby('voter')['round_num'].transform('min')
    
    # Calculate voter statistics
    voter_stats = all_voters.groupby('voter').agg({
        'donated_usd': 'sum',
        'num_projects': 'sum',
        'round_num': 'nunique'
    }).reset_index()
    
    # Calculate global statistics
    stats = {
        'average_voter_value': voter_stats['donated_usd'].mean(),
        'median_voter_value': voter_stats['donated_usd'].median(),
        'total_unique_voters': voter_stats['voter'].nunique(),
        'total_donated_usd': voter_stats['donated_usd'].sum(),
        'average_projects_per_voter': voter_stats['num_projects'].mean(),
        'median_projects_per_voter': voter_stats['num_projects'].median(),
        'average_rounds_per_voter': voter_stats['round_num'].mean(),
        'median_rounds_per_voter': voter_stats['round_num'].median()
    }
    
    # Add donor categorization
    all_voters['donor_wealth_bracket'] = all_voters.apply(
        lambda row: 'minnow' if row['donated_usd'] < 1 
        else 'fish' if row['donated_usd'] < 500 
        else 'whale', 
        axis=1
    )
    
    # Calculate retention data
    all_voters['prev_round'] = all_voters.groupby('voter')['round_num'].shift()
    
    # Create cohort and retention tables
    min_cohort = int(all_voters['cohort'].min())
    max_cohort = int(all_voters['cohort'].max())
    cohort_table = pd.pivot_table(
        all_voters, 
        values='voter', 
        index='cohort',
        columns='round_num',
        aggfunc='count',
        fill_value=0
    )
    
    # Ensure cohort_table includes all cohorts
    all_cohorts = pd.Index(range(min_cohort, max_cohort + 1), name='cohort')
    cohort_table = cohort_table.reindex(all_cohorts, fill_value=0)
    
    # Shift cohort table rows
    for i in range(cohort_table.shape[0]):
        non_zero_index = next((index for index, value in enumerate(cohort_table.iloc[i, :]) if value != 0), None)
        if non_zero_index is not None:
            cohort_table.iloc[i, :] = np.roll(cohort_table.iloc[i, :], -non_zero_index)
    
    # Reset column names and calculate retention
    cohort_table.columns = list(range(cohort_table.shape[1]))
    initial_cohort_sizes = cohort_table.iloc[:, 0].replace(0, 1)
    retention_table = cohort_table.iloc[:, 0:].divide(initial_cohort_sizes, axis=0)
    
    return all_voters, cohort_table, retention_table, stats

def categorize_voter(row):
    if pd.isnull(row['prev_round']) and pd.notnull(row['round_num']):
        return 'new'
    elif pd.notnull(row['prev_round']) and row['prev_round'] == row['round_num'] - 1:
        return 'retained'
    elif pd.notnull(row['prev_round']) and row['prev_round'] < row['round_num'] - 1:
        return 'resurrected'

def plot_voter_counts(all_voters):
    # Create a new dataframe with previous round information
    all_voters['prev_round'] = all_voters.groupby('voter')['round_num'].shift()

    # Apply the function to categorize voters
    all_voters['voter_type'] = all_voters.apply(categorize_voter, axis=1)

    # Group by round_num and voter_type and count the unique voters
    voter_counts_pivot = all_voters.groupby(['round_num', 'voter_type'])['voter'].nunique().unstack()

    # Plot the data
    fig = go.Figure(data=[
        go.Bar(x=voter_counts_pivot.index.astype(str), y=voter_counts_pivot[voter_type], name=voter_type.capitalize()) 
        for voter_type in ['new', 'retained', 'resurrected']
    ])

    # Stack the bars and format the graph
    fig.update_layout(barmode='stack')
    fig = format_graph(fig, 'Donor Participation per Round', 'Round Number', 'Donor Counts')
    # Show the plot
    return(fig)

def plot_retention_rate(retention_table):
    for n in range(len(retention_table)):
        if n != 0:
            retention_table.iloc[n, -n:] = np.nan

    fig = px.imshow(retention_table,
                    labels=dict(x="Round Number", y="Cohort", color="Retention Rate"),
                    x=retention_table.columns[:],
                    y=retention_table.index,
                    color_continuous_scale=px.colors.sequential.Blues,
                    text_auto='.2%',
                    aspect="auto")

    fig.update_layout(
        xaxis=dict(
            title='Retention Rate by Rounds Since Cohort Joined',
            side='top'
        ),
        yaxis_title='Cohort',
        plot_bgcolor='white',
        font=dict(size=12),
        width=800,
        height=800
    )

    # Update x-axis and y-axis to show every number
    fig.update_xaxes(tickmode='linear', dtick=1)
    fig.update_yaxes(tickmode='linear', dtick=1)

    # Show the plot
    return fig

def plot_retention_and_users(cohort_table, retention_table, format_graph):
    # Exclude cohorts 1,2,3
    cohort_table = cohort_table.iloc[3:]
    retention_table = retention_table.iloc[3:]
    
    # Calculate the mean retention for each round
    for n in range(len(retention_table)):
        if n != 0:
            retention_table.iloc[n, -n:] = np.nan
    mean_retention = retention_table.mean()

    # Calculate the number of users for each round
    num_users = [cohort_table.iloc[:-n, 0].sum() if n > 0 else cohort_table.iloc[:, 0].sum() for n in range(len(cohort_table))]

    # Create a figure
    fig = go.Figure()

    # Add mean retention line
    fig.add_trace(go.Scatter(
        x=mean_retention.index, 
        y=mean_retention.values, 
        mode='lines+markers', 
        name='Mean Retention', 
        line=dict(color='royalblue', width=1.2),
        marker=dict(color='royalblue', size=6)
    ))

    # Add number of users line
    fig.add_trace(go.Scatter(
        x=list(range(len(num_users))), 
        y=num_users, 
        mode='lines+markers', 
        name='# of Users', 
        line=dict(color='green', width=1.2),
        marker=dict(color='green', size=6),
        yaxis='y2'
    ))

    # Add a dashed line at y = 0.21
    fig.add_shape(
        type="line",
        x0=0,
        y0=0.125,
        x1=max(mean_retention.index),
        y1=0.125,
        line=dict(
            color="DarkSlateGrey",
            width=1.5,
            dash="dash",
        )
    )

    # Use format_graph function to format the graph
    fig = format_graph(fig, 'Long-Term Donor Retention Rate Flatlines Around 12.5%', 'Rounds Retained', 'Retention Rate', '# of Users')

    # Ensure y-axis starts at zero
    fig.update_yaxes(rangemode="tozero")

    # Show the plot
    return fig

def categorize_donor_by_donation(row):
    if row['donated_usd'] < 1:
        return 'minnow'
    elif 1 <= row['donated_usd'] < 500:
        return 'fish'
    else:
        return 'whale'

def plot_donor_counts(all_voters):
    # Apply the function to categorize donors
    all_voters['donor_wealth_bracket'] = all_voters.apply(categorize_donor_by_donation, axis=1)

    # Group by round_num and donor_wealth_bracket and count the unique donors
    donor_counts_pivot = all_voters.groupby(['round_num', 'donor_wealth_bracket'])['voter'].nunique().unstack()

    # Plot the data
    fig = go.Figure(data=[
        go.Bar(x=donor_counts_pivot.index.astype(str), y=donor_counts_pivot[donor_type], name=donor_type.capitalize()) 
        for donor_type in ['minnow', 'fish',  'whale']
    ])

    # Stack the bars and format the graph
    fig.update_layout(barmode='stack')
    # Calculate the % decrease in Fish from round 18 to 20
    fish_decrease = ((donor_counts_pivot.loc[15, 'fish'] - donor_counts_pivot.loc[20, 'fish']) / donor_counts_pivot.loc[15, 'fish']) * 100

    # Calculate the % increase in Minnows from round 18 to 20
    minnow_increase = ((donor_counts_pivot.loc[20, 'minnow'] - donor_counts_pivot.loc[15, 'minnow']) / donor_counts_pivot.loc[15, 'minnow']) * 100

    # Format the graph with the calculated % decrease and % increase
    fig = format_graph(fig, f'Count of Donors by Category', 'Round Number', 'Donor Counts')
    return fig

def plot_donation_amounts(all_voters):
    # Group by round_num and donor_wealth_bracket and sum the donated_usd
    donation_amounts_pivot = all_voters.groupby(['round_num', 'donor_wealth_bracket'])['donated_usd'].sum().unstack()

    # Plot the data
    fig = go.Figure(data=[
        go.Scatter(x=donation_amounts_pivot.index.astype(str), y=donation_amounts_pivot[donor_type], name=donor_type.capitalize(), stackgroup='one', mode='none') 
        for donor_type in ['minnow', 'fish', 'whale']
    ])

    # Format the graph
    whale_decrease = ((donation_amounts_pivot.loc[15, 'whale'] - donation_amounts_pivot.loc[20, 'whale']) / donation_amounts_pivot.loc[15, 'whale']) * 100
    fish_decrease = ((donation_amounts_pivot.loc[15, 'fish'] - donation_amounts_pivot.loc[20, 'fish']) / donation_amounts_pivot.loc[15, 'fish']) * 100
    fig = format_graph(fig, f'Donations by Donor Category', 'Round Number', 'Donation Amounts (USD)')
    # Show the plot
    return fig

def plot_crowdfunding_trends(all_voters):
    # Extract new and returning users data
    new_users = all_voters[all_voters.groupby('voter')['round_num'].transform('min') == all_voters['round_num']].groupby('round_num')['voter'].nunique()
    returning_users = all_voters[all_voters.groupby('voter')['round_num'].transform('min') != all_voters['round_num']].groupby('round_num')['voter'].nunique()
    crowdfunding_in_usd = all_voters.groupby('round_num')['donated_usd'].sum()

    fig = go.Figure()

    # Add scatter plot for new users
    fig.add_trace(go.Scatter(
        x=new_users.index, 
        y=new_users.values,
        mode='lines+markers',
        marker=dict(size=[9 if x == 15 or x == 20 else 6 for x in new_users.index], color='blue', line=dict(width=1, color='DarkSlateGrey')),
        line=dict(color='blue', width=1.2),
        name='New Voters',
        yaxis='y1'
    ))

    # Add scatter plot for returning users
    fig.add_trace(go.Scatter(
        x=returning_users.index, 
        y=returning_users.values,
        mode='lines+markers',
        marker=dict(size=[9 if x == 15 or x == 20 else 6 for x in returning_users.index], color='red', line=dict(width=1, color='DarkSlateGrey')),
        line=dict(color='red', width=1.2),
        name='Returning Voters',
        yaxis='y1'
    ))

    # Add crowdfunding line on second yaxis
    fig.add_trace(go.Scatter(
        x=crowdfunding_in_usd.index,
        y=crowdfunding_in_usd.values,
        mode='lines+markers',
        marker=dict(size=[9 if x == 15 or x == 20 else 6 for x in crowdfunding_in_usd.index], color='green', line=dict(width=1, color='DarkSlateGrey'), symbol='star'),
        line=dict(color='green', width=1.2),
        name='Crowdfunding in USD',
        yaxis='y2'
    ))

    fig = format_graph(fig, 'Crowdfunding, New, and Returning Voters by Round', 'Round Number', 'Number of Voters', 'Crowdfunding in USD')
    fig.add_vline(x=15, line=dict(color='black', width=2, dash='dash'))
    
    return fig

def plot_funding_by_retention():
    # Calculate the amount of funding given a retention rate between 0 and 1
    retention_rate = np.linspace(0, 1, 100)
    funding = retention_rate * 23.72 * 35000

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=retention_rate, 
        y=funding, 
        mode='lines', 
        name='Funding',
        line=dict(color='blue', width=2)
    ))

    # Add funding amount labels at specific points
    for x_val in [0.2, 0.4, 0.6]:
        fig.add_trace(go.Scatter(
            x=[x_val],
            y=[x_val * 23.72 * 35000],
            mode='markers+text',
            text=[f"${x_val * 23.72 * 35000 / 1000:.2f}k"],
            textposition="top center",
            marker=dict(color='blue', size=8),
            showlegend=False
        ))

    fig = format_graph(fig, 'Contribution to GG21 GMV by Retention Rate', 'Retention Rate', 'Funding')
    return fig

def plot_round_statistics(all_voters):
    # Group by 'round_num' and calculate statistics
    round_stats = all_voters.groupby('round_num').agg({
        'voter': 'nunique',
        'donated_usd': ['mean', 'median']
    }).reset_index()

    round_stats = round_stats[round_stats['round_num'] > 10]
    round_stats.columns = ['round_num', 'unique_voters', 'average_donated_usd', 'median_donated_usd']

    fig = go.Figure()

    # Plot number of unique voters
    fig.add_trace(go.Scatter(
        x=round_stats['round_num'], 
        y=round_stats['unique_voters'], 
        mode='lines+markers', 
        name='Unique Voters',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8),
        yaxis='y1'
    ))

    # Plot average and median donated USD
    fig.add_trace(go.Scatter(
        x=round_stats['round_num'], 
        y=round_stats['average_donated_usd'], 
        mode='lines+markers', 
        name='Average Donated USD',
        line=dict(color='firebrick', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=round_stats['round_num'], 
        y=round_stats['median_donated_usd'], 
        mode='lines+markers', 
        name='Median Donated USD',
        line=dict(color='green', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))

    fig = format_graph(fig, "Voter Statistics by Round Number", "Round Number", "Number of Unique Voters", "Donated USD")
    return fig

def plot_donation_distribution(all_voters):
    # Calculate the total donated amount for each voter
    donor_donations = all_voters.groupby('voter')['donated_usd'].sum()
    donor_donations_sorted = donor_donations.sort_values(ascending=False)
    cumulative_donation = np.cumsum(donor_donations_sorted)
    cumulative_donation_percent = cumulative_donation / cumulative_donation.iloc[-1] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(cumulative_donation_percent))/len(cumulative_donation_percent)*100, 
        y=cumulative_donation_percent, 
        mode='lines',
        name='Cumulative Donation',
        line=dict(color='blue', width=2)
    ))

    # Add percentile lines
    percentiles = [50, 70, 90]
    for percentile in percentiles:
        index = np.searchsorted(cumulative_donation_percent, percentile)
        x_val = index/len(cumulative_donation_percent)*100
        fig.add_shape(
            type="line", 
            x0=x_val, y0=0, 
            x1=x_val, y1=percentile, 
            line=dict(color="Red", width=1, dash="dash")
        )
        fig.add_shape(
            type="line", 
            x0=0, y0=percentile, 
            x1=x_val, y1=percentile, 
            line=dict(color="Red", width=1, dash="dash")
        )

    fig.update_layout(
        title='Cumulative Donation Distribution',
        xaxis_title='Donor Rank (Pct)',
        yaxis_title='% Total Donation',
        autosize=False,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# Main app logic
def main():
    # Load data
    gg_voters, gg22_voters = load_data()
    
    # Process data based on filters
    all_voters, cohort_table, retention_table, stats = process_data(
        gg_voters, 
        gg22_voters, 
        FILTER_TO_FC, 
        FILTER_TO_GS, 
        FILTER_TO_PASSPORT
    )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Retention Analysis", "Donor Analysis", "Distribution"])
    
    with tab1:
        st.header("Overview Statistics - Lifetime")
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unique Voters", f"{stats['total_unique_voters']:,}")
            st.metric("Total Donated USD", f"${stats['total_donated_usd']:,.0f}")
        with col2:
            st.metric("Avg Projects per Voter", f"{stats['average_projects_per_voter']:.2f}")
            st.metric("Median Projects per Voter", f"{stats['median_projects_per_voter']:.2f}")
        with col3:
            st.metric("Avg Voter Value", f"${stats['average_voter_value']:,.2f}")
            st.metric("Median Voter Value", f"${stats['median_voter_value']:,.2f}")
        
        # Overview plots
        st.plotly_chart(plot_crowdfunding_trends(all_voters), use_container_width=True)
        st.plotly_chart(plot_round_statistics(all_voters), use_container_width=True)
        st.plotly_chart(plot_voter_counts(all_voters), use_container_width=True)
    
    with tab2:
        st.header("Retention Analysis")
        st.markdown("**Retention is highly affected by sybils. Suggest filtering to farcaster users when looking at retention data.**")
        
        # Retention plots
        st.plotly_chart(plot_retention_rate(retention_table), use_container_width=True)
        st.plotly_chart(plot_retention_and_users(cohort_table, retention_table, format_graph), use_container_width=True)
        #st.plotly_chart(plot_funding_by_retention(), use_container_width=True)
        
        # Add explanation text
        st.markdown("""
        ### Key Retention Insights:
        - Long-term retention rate stabilizes around 12.5%
        - New donor acquisition has declined since Round 15
        - Retention patterns vary significantly by donor size
        """)
    
    with tab3:
        st.header("Donor Analysis")
                # Add explanation text
        st.markdown("""
        ### Donor Categories:
        - **Minnow**: < $1 USD
        - **Fish**: $1-500 USD
        - **Whale**: > $500 USD
        """)
        # Donor analysis plots
        st.plotly_chart(plot_donor_counts(all_voters), use_container_width=True)
        st.plotly_chart(plot_donation_amounts(all_voters), use_container_width=True)
        

    
    with tab4:
        st.header("Distribution Analysis")
        
        # Distribution plots
        st.plotly_chart(plot_donation_distribution(all_voters), use_container_width=True)
        # Calculate and display some distribution statistics
        donor_donations = all_voters.groupby('voter')['donated_usd'].sum()
        donor_donations_sorted = donor_donations.sort_values(ascending=False)
        target_contributions = [0.5, 0.7, 0.9] # 50%, 70%, 90% of total contributions
        
        total_donations = donor_donations.sum()
        
        for target in target_contributions:
            # Find what percentage of donors contribute this much
            cumsum = donor_donations_sorted.cumsum()
            donor_pct = (cumsum < (target * total_donations)).sum() / len(donor_donations)
            min_donation = donor_donations_sorted.iloc[int(donor_pct * len(donor_donations))]
            st.markdown(f"""
            ### {target:.0%} of donations come from top {donor_pct:.1%} of donors (min donation: ${min_donation:,.2f})
            """)

if __name__ == "__main__":
    main()