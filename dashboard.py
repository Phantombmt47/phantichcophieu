import streamlit as st
import pandas as pd
from vnstock import Finance
import plotly.graph_objects as go
import plotly.express as px
import warnings

# Ignore warnings from vnstock or pandas
warnings.filterwarnings("ignore")

# Streamlit page configuration
st.set_page_config(
    page_title="Dashboard Phân tích Báo cáo Tài chính Cổ phiếu Việt Nam (vnstock)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🧠 DATA LOADING FUNCTION (WITH CACHE)
# ==============================
@st.cache_data(ttl=3600, show_spinner=False)
def load_finance_data(symbol, period='year', lang='vi'):
    """
    Loads financial data for a given stock symbol.

    Args:
        symbol (str): The stock symbol.
        period (str): The period for the financial data ('year' or 'quarter').
        lang (str): The language for the financial statements ('vi or 'en').

    Returns:
        dict: A dictionary containing DataFrames for income_statement, cash_flow, and ratios.
              Returns empty Dataframes in case of an error.
    """
    try:
        finance = Finance(symbol=symbol, source='TCBS')

        # Fetch data. Handle potential errors or empty returns from the API.
        balance_sheet_df = finance.balance_sheet(period=period, lang=lang, dropna=False) # Keep NA for now
        income_statement_df = finance.income_statement(period=period, lang=lang, dropna=False) # Keep NA for now
        cash_flow_df = finance.cash_flow(period=period, dropna=False) # Keep NA for now
        ratios_df = finance.ratio(period=period, lang=lang, dropna=False) # Keep NA for now


        # Ensure all are DataFrames and not None. If None, create empty DataFrame.
        balance_sheet_df = balance_sheet_df if isinstance(balance_sheet_df, pd.DataFrame) else pd.DataFrame()
        income_statement_df = income_statement_df if isinstance(income_statement_df, pd.DataFrame) else pd.DataFrame()
        cash_flow_df = cash_flow_df if isinstance(cash_flow_df, pd.DataFrame) else pd.DataFrame()
        ratios_df = ratios_df if isinstance(ratios_df, pd.DataFrame) else pd.DataFrame()


        # Sort dataframes by index (period) in ascending order for consistent charting
        # This sorting logic is added here to ensure consistency regardless of data source order
        # Only sort if the dataframe is not empty
        if not balance_sheet_df.empty:
            balance_sheet_df = balance_sheet_df.sort_index(ascending=True)
        if not income_statement_df.empty:
            income_statement_df = income_statement_df.sort_index(ascending=True)
        if not cash_flow_df.empty:
            cash_flow_df = cash_flow_df.sort_index(ascending=True)
        if not ratios_df.empty:
            ratios_df = ratios_df.sort_index(ascending=True)


        return {
            "balance_sheet": balance_sheet_df,
            "income_statement": income_statement_df,
            "cash_flow": cash_flow_df,
            "ratios": ratios_df
        }

    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return {
            "balance_sheet": pd.DataFrame(),
            "income_statement": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
            "ratios": pd.DataFrame()
        }

def create_grid_charts(revenue_profit_df, cash_flow_df_full, asset_structure_df, capital_structure_df, profit_margins_df, efficiency_df, liquidity_df, leverage_df, pe_ratio_df, pb_ratio_df, working_capital_df, free_cash_flow_df, latest_period):
    """
    Creates financial charts for the dashboard grid.

    Args:
        revenue_profit_df (pd.DataFrame): DataFrame for Revenue & Profit.
        cash_flow_df_full (pd.DataFrame): DataFrame containing all Cash Flow types.
        asset_structure_df (pd.DataFrame): DataFrame for Asset Structure.
        capital_structure_df (pd.DataFrame): DataFrame for Capital Structure.
        profit_margins_df (pd.DataFrame): DataFrame for Profit Margins.
        efficiency_df (pd.DataFrame): DataFrame for ROA & ROE.
        liquidity_df (pd.DataFrame): DataFrame for Liquidity Ratios.
        leverage_df (pd.DataFrame): DataFrame for Leverage Ratio.
        pe_ratio_df (pd.DataFrame): DataFrame for P/E Ratio.
        pb_ratio_df (pd.DataFrame): DataFrame for P/B Ratio.
        working_capital_df (pd.DataFrame): DataFrame for Working Capital.
        free_cash_flow_df (pd.DataFrame): DataFrame for Free Cash Flow.
        latest_period (str): The label for the latest period.


    Returns:
        dict: A dictionary containing Plotly chart objects.
    """
    charts = {}

    # 1. Revenue & Profit (Bar Chart)
    if not revenue_profit_df.empty:
        fig = go.Figure(data=[
            go.Bar(name='Doanh thu', x=revenue_profit_df.index, y=revenue_profit_df['revenue']),
            go.Bar(name='Lợi nhuận sau thuế', x=revenue_profit_df.index, y=revenue_profit_df['post_tax_profit'])
        ])
        fig.update_layout(barmode='group', title='Doanh thu & Lợi nhuận sau thuế (Tỷ đồng)')
        charts['revenue_profit_chart'] = fig
    else:
        charts['revenue_profit_chart'] = None


    # 2. Cash Flow (Bar Chart - all three types)
    cash_flow_cols = ['from_sale', 'from_invest', 'from_financial']
    # Check if cash_flow_df_full is not empty and all required columns exist and have at least one non-NaN value
    if not cash_flow_df_full.empty and all(col in cash_flow_df_full.columns and not cash_flow_df_full[col].dropna().empty for col in cash_flow_cols):
        fig = go.Figure(data=[
            go.Bar(name='Từ HĐKD', x=cash_flow_df_full.index, y=cash_flow_df_full['from_sale']),
            go.Bar(name='Từ HĐĐT', x=cash_flow_df_full.index, y=cash_flow_df_full['from_invest']),
            go.Bar(name='Từ HĐTC', x=cash_flow_df_full.index, y=cash_flow_df_full['from_financial'])
        ])
        fig.update_layout(barmode='group', title='Dòng tiền từ các Hoạt động (Tỷ đồng)')
        charts['cash_flow_chart'] = fig
    else:
        charts['cash_flow_chart'] = None


    # 3. Asset Structure (Pie Chart - Latest Year/Quarter)
    if not asset_structure_df.empty:
        fig = px.pie(asset_structure_df, values='Value', names='Category', title=f'Cơ cấu Tài sản ({latest_period if latest_period else "N/A"})')
        charts['asset_structure_chart'] = fig
    else:
        charts['asset_structure_chart'] = None

    # 4. Profit Margins (Line Chart)
    if not profit_margins_df.empty:
        fig = go.Figure()
        # Only add traces if the column exists and has non-NaN values
        if 'gross_profit_margin' in profit_margins_df.columns and not profit_margins_df['gross_profit_margin'].dropna().empty:
            fig.add_trace(go.Scatter(x=profit_margins_df.index, y=profit_margins_df['gross_profit_margin'], mode='lines+markers', name='Tỷ suất Lợi nhuận Gộp'))
        if 'net_profit_margin' in profit_margins_df.columns and not profit_margins_df['net_profit_margin'].dropna().empty:
            fig.add_trace(go.Scatter(x=profit_margins_df.index, y=profit_margins_df['net_profit_margin'], mode='lines+markers', name='Tỷ suất Lợi nhuận Ròng'))

        # Check if any traces were added before assigning to charts
        if fig.data:
            fig.update_layout(title='Các Tỷ suất Lợi nhuận (%)')
            charts['profit_margins_chart'] = fig
        else:
             charts['profit_margins_chart'] = None # Set to None if no data to plot
    else:
        charts['profit_margins_chart'] = None


    # 5. Efficiency: ROA & ROE (Line Chart)
    if not efficiency_df.empty:
        fig = go.Figure()
        # Only add traces if the column exists and has non-NaN values
        if 'roa' in efficiency_df.columns and not efficiency_df['roa'].dropna().empty:
            fig.add_trace(go.Scatter(x=efficiency_df.index, y=efficiency_df['roa'], mode='lines+markers', name='ROA'))
        if 'roe' in efficiency_df.columns and not efficiency_df['roe'].dropna().empty:
            fig.add_trace(go.Scatter(x=efficiency_df.index, y=efficiency_df['roe'], mode='lines+markers', name='ROE'))

        # Check if any traces were added before assigning to charts
        if fig.data:
            fig.update_layout(title='Hiệu quả Hoạt động (ROA & ROE - %)')
            fig.update_layout(yaxis_tickformat=".2%") # Format as percentage
            charts['efficiency_chart'] = fig
        else:
            charts['efficiency_chart'] = None # Set to None if no data to plot
    else:
        charts['efficiency_chart'] = None

    # 6. Capital Structure (Pie Chart - Latest Year/Quarter)
    if not capital_structure_df.empty:
        fig = px.pie(capital_structure_df, values='Value', names='Category', title=f'Cơ cấu Nguồn vốn ({latest_period if latest_period else "N/A"})')
        charts['capital_structure_chart'] = fig
    else:
        charts['capital_structure_chart'] = None

    # 7. Liquidity: Current & Quick Ratios (Line Chart)
    if not liquidity_df.empty:
        fig = go.Figure()
        if 'current_ratio' in liquidity_df.columns and not liquidity_df['current_ratio'].dropna().empty:
             fig.add_trace(go.Scatter(x=liquidity_df.index, y=liquidity_df['current_ratio'], mode='lines+markers', name='Tỷ lệ Thanh toán Hiện hành'))
        if 'quick_ratio' in liquidity_df.columns and not liquidity_df['quick_ratio'].dropna().empty:
             fig.add_trace(go.Scatter(x=liquidity_df.index, y=liquidity_df['quick_ratio'], mode='lines+markers', name='Tỷ lệ Thanh toán Nhanh'))
        fig.update_layout(title='Các Tỷ lệ Thanh khoản')
        # Check if any traces were added before assigning to charts
        if fig.data:
            charts['liquidity_chart'] = fig
        else:
            charts['liquidity_chart'] = None # Set to None if no data to plot
    else:
        charts['liquidity_chart'] = None

    # 8. Leverage Ratio (Debt/Equity) (Line Chart)
    if not leverage_df.empty and 'debt_on_equity' in leverage_df.columns and not leverage_df['debt_on_equity'].dropna().empty:
        fig = px.line(leverage_df, x=leverage_df.index, y='debt_on_equity', title='Tỷ lệ Đòn bẩy Tài chính (Nợ/Vốn Chủ sở hữu)')
        charts['leverage_chart'] = fig
    else:
        charts['leverage_chart'] = None

    # 9. P/E Ratio (Line Chart)
    if not pe_ratio_df.empty and 'price_to_earning' in pe_ratio_df.columns and not pe_ratio_df['price_to_earning'].dropna().empty:
        fig = px.line(pe_ratio_df, x=pe_ratio_df.index, y='price_to_earning', title='Chỉ số P/E')
        charts['pe_ratio_chart'] = fig
    else:
        charts['pe_ratio_chart'] = None

    # 10. P/B Ratio (Line Chart)
    if not pb_ratio_df.empty and 'price_to_book' in pb_ratio_df.columns and not pb_ratio_df['price_to_book'].dropna().empty:
        fig = px.line(pb_ratio_df, x=pb_ratio_df.index, y='price_to_book', title='Chỉ số P/B')
        charts['pb_ratio_chart'] = fig
    else:
        charts['pb_ratio_chart'] = None

    # 11. Working Capital Trend (Line Chart)
    if not working_capital_df.empty and 'working_capital' in working_capital_df.columns and not working_capital_df['working_capital'].dropna().empty:
        fig = px.line(working_capital_df, x=working_capital_df.index, y='working_capital', title='Xu hướng Vốn lưu động (Tỷ đồng)')
        charts['working_capital_chart'] = fig
    else:
        charts['working_capital_chart'] = None

    # 12. Free Cash Flow Trend (Line Chart)
    if not free_cash_flow_df.empty and 'free_cash_flow' in free_cash_flow_df.columns and not free_cash_flow_df['free_cash_flow'].dropna().empty:
        fig = px.line(free_cash_flow_df, x=free_cash_flow_df.index, y='free_cash_flow', title='Xu hướng Dòng tiền tự do (Tỷ đồng)')
        charts['free_cash_flow_chart'] = fig
    else:
        charts['free_cash_flow_chart'] = None


    return charts


# ==============================
# 🧱 MAIN INTERFACE
# ==============================

st.title("📊 Dashboard Phân tích Báo cáo Tài chính Cổ phiếu Việt Nam")

# Sidebar stock selection
st.sidebar.header("🔍 Lựa chọn Cổ phiếu")

# Use st.text_input for user to enter any stock symbol
selected_symbol = st.sidebar.text_input(
    "Nhập Mã Cổ phiếu:",
    value='TNG' # Default value
).upper() # Convert to uppercase to ensure consistency

# Add period selection to the sidebar
selected_period = st.sidebar.selectbox(
    "Chọn Kỳ báo cáo:",
    options=['year', 'quarter'],
    index=0, # Default to 'year'
    format_func=lambda x: 'Năm' if x == 'year' else 'Quý' # Display in Vietnamese
)


st.header(f"Báo cáo tài chính cho {selected_symbol} ({'Năm' if selected_period == 'year' else 'Quý'})")

# Load data for the selected symbol and period
finance_data_dict = load_finance_data(selected_symbol, period=selected_period)

# Access dataframes from the dictionary
balance_sheet_df = finance_data_dict.get("balance_sheet", pd.DataFrame())
income_statement_df = finance_data_dict.get("income_statement", pd.DataFrame())
cash_flow_df = finance_data_dict.get("cash_flow", pd.DataFrame())
ratios_df = finance_data_dict.get("ratios", pd.DataFrame())

# Define KPIs and their formatting
kpis = [
    'revenue',
    'post_tax_profit',
    'roe',
    'roa',
    'debt_on_equity',
    'price_to_earning',
    'price_to_book'
]

# Vietnamese labels and formatting for KPIs
kpi_labels = {
    'revenue': 'Doanh thu',
    'post_tax_profit': 'Lợi nhuận sau thuế',
    'roe': 'ROE',
    'roa': 'ROA',
    'debt_on_equity': 'Nợ/Vốn CSH',
    'price_to_earning': 'P/E',
    'price_to_book': 'P/B'
}

kpi_formats = {
    'revenue': '{:,.0f}',
    'post_tax_profit': '{:,.0f}',
    'roe': '{:.2%}',
    'roa': '{:.2%}',
    'debt_on_equity': '{:.2f}',
    'price_to_earning': '{:.2f}',
    'price_to_book': '{:.2f}'
}


if finance_data_dict and any(not df.empty for df in finance_data_dict.values()):
    # Identify the latest period from all available dataframes
    latest_period = None
    all_indices = []
    if not balance_sheet_df.empty: all_indices.extend(balance_sheet_df.index.tolist())
    if not income_statement_df.empty: all_indices.extend(income_statement_df.index.tolist())
    if not cash_flow_df.empty: all_indices.extend(cash_flow_df.index.tolist())
    if not ratios_df.empty: all_indices.extend(ratios_df.index.tolist())

    if all_indices:
        # Assuming indices are sortable (e.g., years or quarters like "2023Q4")
        latest_period = sorted(list(set(all_indices)))[-1]


    st.subheader(f"Các chỉ số chính (KPIs) cho kỳ gần nhất ({latest_period if latest_period else 'N/A'})")

    if latest_period is not None:
        # Extract KPI values for the latest period
        kpi_values = {}
        for kpi in kpis:
            value = None
            # Check income statement first
            if not income_statement_df.empty and kpi in income_statement_df.columns and latest_period in income_statement_df.index:
                value = income_statement_df.loc[latest_period, kpi]
            # If not found in income statement, check ratios
            elif not ratios_df.empty and kpi in ratios_df.columns and latest_period in ratios_df.index:
                value = ratios_df.loc[latest_period, kpi]
            kpi_values[kpi] = value

        # Display KPIs using columns
        cols = st.columns(len(kpis))
        for i, kpi in enumerate(kpis):
            with cols[i]:
                label = kpi_labels.get(kpi, kpi)
                value = kpi_values.get(kpi)
                if value is not None and pd.notna(value): # Check for None and NaN
                    # Apply specific formatting
                    if kpi in ['roe', 'roa']:
                         formatted_value = kpi_formats.get(kpi, '{}').format(value)
                    elif kpi in ['revenue', 'post_tax_profit']:
                         # Handle potential non-numeric data gracefully
                         try:
                             formatted_value = kpi_formats.get(kpi, '{}').format(value)
                         except (ValueError, TypeError):
                             formatted_value = "N/A"
                    elif pd.notna(value):
                         formatted_value = kpi_formats.get(kpi, '{}').format(value)
                    else:
                         formatted_value = "N/A"

                    st.metric(label=label, value=formatted_value)
                else:
                    st.metric(label=label, value="N/A")


    else:
        st.info("Không có dữ liệu để hiển thị chỉ số chính.")

    st.subheader("Xu hướng tài chính theo thời gian và Cơ cấu")

    # Prepare data for grid charts *inside* the data check block
    revenue_profit_df = income_statement_df[['revenue', 'post_tax_profit']].copy() if 'revenue' in income_statement_df.columns and 'post_tax_profit' in income_statement_df.columns else pd.DataFrame()
    # Include all three cash flow types
    cash_flow_df_full = cash_flow_df[['from_sale', 'from_invest', 'from_financial']].copy() if all(col in cash_flow_df.columns for col in ['from_sale', 'from_invest', 'from_financial']) else pd.DataFrame()


    # Asset & Capital Structure (Most Recent Year/Quarter)
    asset_structure_df = pd.DataFrame()
    capital_structure_df = pd.DataFrame()
    if latest_period is not None and not balance_sheet_df.empty and latest_period in balance_sheet_df.index:
        latest_balance_sheet = balance_sheet_df.loc[latest_period] # Corrected typo here
        # Ensure required columns exist and are not NaN before creating the structure
        # Use .get() with a default of NaN to avoid KeyError if column is missing
        short_asset_val = latest_balance_sheet.get('short_asset', pd.NA)
        long_asset_val = latest_balance_sheet.get('long_asset', pd.NA)
        debt_val = latest_balance_sheet.get('debt', pd.NA)
        equity_val = latest_balance_sheet.get('equity', pd.NA)


        if pd.notna(short_asset_val) and pd.notna(long_asset_val):
            asset_data = {'Category': ['Tài sản ngắn hạn', 'Tài sản dài hạn'],
                          'Value': [short_asset_val, long_asset_val]}
            asset_structure_df = pd.DataFrame(asset_data)


        if pd.notna(debt_val) and pd.notna(equity_val):
             capital_data = {'Category': ['Nợ phải trả', 'Vốn chủ sở hữu'],
                            'Value': [debt_val, equity_val]}
             capital_structure_df = pd.DataFrame(capital_data)


    # Profit Margins
    profit_margins_df = pd.DataFrame()
    if not income_statement_df.empty:
        # Ensure required columns exist
        if 'gross_profit' in income_statement_df.columns and 'revenue' in income_statement_df.columns and 'post_tax_profit' in income_statement_df.columns:
            # Avoid division by zero or missing values in revenue
            # Use .copy() to avoid SettingWithCopyWarning
            income_statement_df_cleaned = income_statement_df.copy()
            # Replace 0 or NA in revenue with NA for division, affecting only the revenue column for this calculation
            income_statement_df_cleaned['revenue'] = income_statement_df_cleaned['revenue'].replace([0], pd.NA)


            if not income_statement_df_cleaned['revenue'].isna().all(): # Check if there's at least one non-NA revenue value
                 # Calculate margins, handling potential NA/infinity results
                 with pd.option_context('mode.use_inf_as_na', True): # Treat infinity as NA
                     profit_margins_df['gross_profit_margin'] = (income_statement_df_cleaned['gross_profit'] / income_statement_df_cleaned['revenue'])
                     profit_margins_df['net_profit_margin'] = (income_statement_df_cleaned['post_tax_profit'] / income_statement_df_cleaned['revenue'])

                 # Ensure index is string type for plotting, only if profit_margins_df is not empty
                 if not profit_margins_df.empty:
                     profit_margins_df.index = income_statement_df_cleaned.index.astype(str)

                 # Explicitly replace infinite values with NaN after calculation
                 profit_margins_df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
            else:
                # Only show warning if revenue column exists but is all NA/0
                if 'revenue' in income_statement_df.columns:
                    st.warning(f"Không đủ dữ liệu doanh thu ({selected_symbol}) để tính Tỷ suất Lợi nhuận.")
        else:
             st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Tỷ suất Lợi nhuận.")


    # Efficiency: ROA & ROE
    efficiency_df = pd.DataFrame()
    if not ratios_df.empty:
        # Select columns only if they exist
        cols_to_select = [col for col in ['roa', 'roe'] if col in ratios_df.columns]
        if cols_to_select:
             efficiency_df = ratios_df[cols_to_select].copy()
             efficiency_df.index = efficiency_df.index.astype(str)
        else:
             st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Hiệu quả Hoạt động (ROA & ROE).")


    # Liquidity: Current & Quick Ratios
    liquidity_df = pd.DataFrame()
    if not balance_sheet_df.empty:
        # Ensure required columns exist before attempting calculations
        required_cols_current = ['short_asset', 'short_debt']
        required_cols_quick = ['cash', 'short_invest', 'short_receivable', 'short_debt']

        can_calc_current = all(col in balance_sheet_df.columns for col in required_cols_current)
        can_calc_quick = all(col in balance_sheet_df.columns for col in required_cols_quick)

        if can_calc_current or can_calc_quick:
            balance_sheet_df_cleaned = balance_sheet_df.copy()
             # Replace 0 or NA in 'short_debt' with NA for division, affecting only the 'short_debt' column
            balance_sheet_df_cleaned['short_debt'] = balance_sheet_df_cleaned['short_debt'].replace([0], pd.NA)


            # Calculate Current Ratio
            current_ratio_calculated = pd.Series(dtype='float64') # Initialize as empty Series
            if can_calc_current and not balance_sheet_df_cleaned['short_debt'].isna().all(): # Added check for non-NA short_debt
                 with pd.option_context('mode.use_inf_as_na', True): # Treat infinity as NA
                      current_ratio_calculated = (balance_sheet_df_cleaned['short_asset'] / balance_sheet_df_cleaned['short_debt'])
            elif can_calc_current: # short_debt column exists but is all NA/0
                st.warning(f"Không đủ dữ liệu ({selected_symbol}) để tính Tỷ lệ Thanh toán Hiện hành ('short_debt' toàn NaN/0).")
            elif required_cols_current[0] in balance_sheet_df.columns and required_cols_current[1] in balance_sheet_df.columns: # Should not happen based on can_calc_current
                 pass # Avoid duplicate warning if columns exist but are all NA/0
            else: # Columns are missing
                 st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Tỷ lệ Thanh toán Hiện hành (Thiếu 'short_asset' hoặc 'short_debt').")


            # Calculate Quick Ratio
            quick_ratio_calculated = pd.Series(dtype='float64') # Initialize as empty Series
            if can_calc_quick and not balance_sheet_df_cleaned['short_debt'].isna().all(): # Added check for non-NA short_debt
                 with pd.option_context('mode.use_inf_as_na', True): # Treat infinity as NA
                      quick_ratio_calculated = ((balance_sheet_df_cleaned['cash'] + balance_sheet_df_cleaned['short_invest'] + balance_sheet_df_cleaned['short_receivable']) / balance_sheet_df_cleaned['short_debt'])
            elif can_calc_quick: # short_debt column exists but is all NA/0
                st.warning(f"Không đủ dữ liệu ({selected_symbol}) để tính Tỷ lệ Thanh toán Nhanh ('short_debt' toàn NaN/0).")
            elif all(col in balance_sheet_df.columns for col in required_cols_quick): # Should not happen based on can_calc_quick
                 pass # Avoid duplicate warning if columns exist but are all NA/0
            else: # Columns are missing
                st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Tỷ lệ Thanh toán Nhanh.")


            # Combine calculated ratios into liquidity_df, ensuring index alignment
            # Use concat to combine, handling potential empty series and aligning by index
            all_ratios_series = [s for s in [current_ratio_calculated.rename('current_ratio'), quick_ratio_calculated.rename('quick_ratio')] if not s.empty]
            if all_ratios_series:
                 liquidity_df = pd.concat(all_ratios_series, axis=1)
                 # Ensure index is string type for plotting, only if liquidity_df is not empty
                 if not liquidity_df.empty:
                      liquidity_df.index = liquidity_df.index.astype(str)
            else:
                 # If no ratios could be calculated but balance sheet data exists, create an empty df with matching index
                 if not balance_sheet_df.empty:
                     liquidity_df = pd.DataFrame(index=balance_sheet_df.index.astype(str))
                 else:
                     liquidity_df = pd.DataFrame() # Truly empty if no balance sheet data

        else:
             st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Tỷ lệ Thanh khoản.")

    # Replace infinite values with NaN after calculation
    liquidity_df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)


    # Calculate Working Capital and Free Cash Flow *after* loading and sorting, but *inside* the data check
    # This ensures calculation is based on the correct, sorted data and only happens if data is available
    # Calculate Working Capital
    working_capital_df = pd.DataFrame()
    if not balance_sheet_df.empty and 'short_asset' in balance_sheet_df.columns and 'short_debt' in balance_sheet_df.columns:
        working_capital_df['working_capital'] = balance_sheet_df['short_asset'] - balance_sheet_df['short_debt']
        working_capital_df.index = balance_sheet_df.index.astype(str) # Corrected index assignment
    elif not balance_sheet_df.empty: # Check if balance sheet is not empty but columns are missing
        st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Vốn lưu động (Thiếu 'short_asset' hoặc 'short_debt').")


    # Calculate Free Cash Flow
    free_cash_flow_df = pd.DataFrame()
    if not cash_flow_df.empty and 'from_sale' in cash_flow_df.columns and 'invest_cost' in cash_flow_df.columns:
        free_cash_flow_df['free_cash_flow'] = cash_flow_df['from_sale'] + cash_flow_df['invest_cost'] # invest_cost is typically negative
        free_cash_flow_df.index = cash_flow_df.index.astype(str)
    elif not cash_flow_df.empty: # Check if cash flow is not empty but columns are missing
         st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Dòng tiền tự do (Thiếu 'from_sale' hoặc 'invest_cost').")



    # Leverage Ratio (Debt/Equity)
    leverage_df = pd.DataFrame()
    if not ratios_df.empty and 'debt_on_equity' in ratios_df.columns:
        leverage_df = ratios_df[['debt_on_equity']].copy()
        leverage_df.index = leverage_df.index.astype(str)
    elif not ratios_df.empty: # Check if ratios is not empty but column is missing
         st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Tỷ lệ Đòn bẩy Tài chính (Thiếu 'debt_on_equity').")


    # P/E Ratio
    pe_ratio_df = pd.DataFrame()
    if not ratios_df.empty and 'price_to_earning' in ratios_df.columns:
        pe_ratio_df = ratios_df[['price_to_earning']].copy()
        pe_ratio_df.index = pe_ratio_df.index.astype(str)
    elif not ratios_df.empty: # Check if ratios is not empty but column is missing
         st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Chỉ số P/E (Thiếu 'price_to_earning').")


    # P/B Ratio
    pb_ratio_df = pd.DataFrame()
    if not ratios_df.empty and 'price_to_book' in ratios_df.columns:
        pb_ratio_df = ratios_df[['price_to_book']].copy()
        pb_ratio_df.index = pb_ratio_df.index.astype(str)
    elif not ratios_df.empty: # Check if ratios is not empty but column is missing
         st.warning(f"Thiếu cột cần thiết ({selected_symbol}) để tính Chỉ số P/B (Thiếu 'price_to_book').")


    # Create the grid charts *inside* the data check block
    grid_charts = create_grid_charts(revenue_profit_df, cash_flow_df_full, asset_structure_df, capital_structure_df, profit_margins_df, efficiency_df, liquidity_df, leverage_df, pe_ratio_df, pb_ratio_df, working_capital_df, free_cash_flow_df, latest_period)


    # Display the charts in a 3x3 grid *inside* the data check block
    col1, col2, col3 = st.columns(3)

    # Row 1
    with col1:
        if grid_charts.get('revenue_profit_chart') is not None:
            st.plotly_chart(grid_charts['revenue_profit_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Doanh thu & Lợi nhuận.")

    with col2:
        if grid_charts.get('cash_flow_chart') is not None:
            st.plotly_chart(grid_charts['cash_flow_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Dòng tiền từ các Hoạt động.")

    with col3:
        if grid_charts.get('asset_structure_chart') is not None:
             st.plotly_chart(grid_charts['asset_structure_chart'], use_container_width=True)
        else:
             st.info("Không có dữ liệu để hiển thị biểu đồ Cơ cấu Tài sản.")


    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        if grid_charts.get('profit_margins_chart') is not None:
            st.plotly_chart(grid_charts['profit_margins_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Các Tỷ suất Lợi nhuận.")

    with col5:
        if grid_charts.get('efficiency_chart') is not None:
            st.plotly_chart(grid_charts['efficiency_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Hiệu quả Hoạt động.")

    with col6:
        if grid_charts.get('capital_structure_chart') is not None:
            st.plotly_chart(grid_charts['capital_structure_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Cơ cấu Nguồn vốn.")


    # Row 3
    col7, col8, col9 = st.columns(3)
    with col7:
         # Display Liquidity chart here
         if grid_charts.get('liquidity_chart') is not None:
             st.plotly_chart(grid_charts['liquidity_chart'], use_container_width=True)
         else:
             st.info("Không có dữ liệu để hiển thị biểu đồ Các Tỷ lệ Thanh khoản.")

    with col8:
        if grid_charts.get('leverage_chart') is not None:
            st.plotly_chart(grid_charts['leverage_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Tỷ lệ Đòn bẩy Tài chính.")

    with col9:
        if grid_charts.get('pe_ratio_chart') is not None:
            st.plotly_chart(grid_charts['pe_ratio_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Chỉ số P/E.")


    # Row 4 (for the two new charts and P/B)
    col10, col11, col12 = st.columns(3)
    with col10:
        if grid_charts.get('working_capital_chart') is not None:
            st.plotly_chart(grid_charts['working_capital_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Xu hướng Vốn lưu động.")

    with col11:
        if grid_charts.get('free_cash_flow_chart') is not None:
            st.plotly_chart(grid_charts['free_cash_flow_chart'], use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ Xu hướng Dòng tiền tự do.")

    with col12:
         if grid_charts.get('pb_ratio_chart') is not None:
             st.plotly_chart(grid_charts['pb_ratio_chart'], use_container_width=True)
         else:
              st.info("Không có dữ liệu để hiển thị biểu đồ Chỉ số P/B.")


    st.subheader("Bảng dữ liệu thô")

    # Display raw data tables
    if not balance_sheet_df.empty:
        st.write("Bảng cân đối kế toán:")
        st.dataframe(balance_sheet_df)
    else:
        st.info("Không có dữ liệu Bảng cân đối kế toán.")

    if not income_statement_df.empty:
        st.write("Báo cáo kết quả kinh doanh:")
        st.dataframe(income_statement_df)
    else:
        st.info("Không có dữ liệu Báo cáo kết quả kinh doanh.")

    if not cash_flow_df.empty:
        st.write("Báo cáo lưu chuyển tiền tệ:")
        st.dataframe(cash_flow_df)
    else:
        st.info("Không có dữ liệu Báo cáo lưu chuyển tiền tệ.")

    if not ratios_df.empty:
        st.write("Các tỷ lệ tài chính:")
        st.dataframe(ratios_df)
    else:
        st.info("Không có dữ liệu Các tỷ lệ tài chính.")

else:
    st.info(f"Không có dữ liệu tài chính cho {selected_symbol} theo kỳ {'Năm' if selected_period == 'year' else 'Quý'}")
