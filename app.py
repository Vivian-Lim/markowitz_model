from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go

app = Flask(__name__)

# Home route with default tickers
@app.route('/')
def home():
    # Default stock tickers
    default_stocks = "AAPL,KO,AMZN,GLD"
    return render_template('index.html', default_stocks=default_stocks)

# Analyze route to handle the form submission
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get stock tickers from user input
    stock_symbols = request.form['stocks'].split(',')

    # Fetch stock data using yfinance
    data = yf.download(stock_symbols, start='2023-01-01', end='2024-01-01')['Adj Close']

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Optimize portfolio (using Markowitz model)
    fig, table_data = optimize_portfolio(returns, stock_symbols)

    # Generate table HTML
    table_html = table_data.to_html(index=False, classes="table table-striped", justify="left")

    # Render the result page with the graph and table
    return render_template('index.html', graph_html=fig.to_html(full_html=False), table_html=table_html, default_stocks=request.form['stocks'])

# Function to optimize portfolio (Markowitz model)
def optimize_portfolio(returns, stock_symbols):
    num_portfolios = 5000
    expected_returns = returns.mean()
    cov_matrix = returns.cov()

    portfolio_results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        portfolio_results[0, i] = portfolio_return
        portfolio_results[1, i] = portfolio_volatility

        if portfolio_volatility != 0:
            portfolio_results[2, i] = portfolio_return / portfolio_volatility
        else:
            portfolio_results[2, i] = np.nan

    # Convert to DataFrame
    results_df = pd.DataFrame(portfolio_results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])
    weights_df = pd.DataFrame(weights_record, columns=stock_symbols)

    # Filter out NaN values and zero volatility
    results_df = results_df.dropna(subset=['Sharpe Ratio'])
    results_df = results_df[results_df['Volatility'] > 0]

    if not results_df.empty:
        max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
        max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]
        max_sharpe_weights = weights_df.iloc[max_sharpe_idx]

        min_vol_idx = results_df['Volatility'].idxmin()
        min_vol_portfolio = results_df.iloc[min_vol_idx]
        min_vol_weights = weights_df.iloc[min_vol_idx]

        # Create table data with stock information
        table_data = pd.DataFrame({
            'Stock': stock_symbols,
            'Expected Return (%)': np.round(expected_returns * 100, 2),
            'Volatility (Risk) (%)': np.round(returns.std() * 100, 2),
            'Max Sharpe Weights (%)': np.round(max_sharpe_weights * 100, 2),
            'Min Volatility Weights (%)': np.round(min_vol_weights * 100, 2)
        })

        # Plot the efficient frontier
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=results_df['Volatility'],
            y=results_df['Return'],
            mode='markers',
            marker=dict(color=results_df['Sharpe Ratio'], showscale=True),
            name="Portfolios"
        ))

        # Add Max Sharpe Ratio point
        fig.add_trace(go.Scatter(
            x=[max_sharpe_portfolio['Volatility']],
            y=[max_sharpe_portfolio['Return']],
            mode='markers',
            marker=dict(color='red', size=12, symbol='circle'),
            name="Max Sharpe Ratio"
        ))

        # Add Min Volatility point
        fig.add_trace(go.Scatter(
            x=[min_vol_portfolio['Volatility']],
            y=[min_vol_portfolio['Return']],
            mode='markers',
            marker=dict(color='blue', size=12, symbol='circle'),
            name="Min Volatility"
        ))

        # Update layout to move the legend to the bottom
        fig.update_layout(
            title="Efficient Frontier (Markowitz Model) with Maximum Sharpe Ratio",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,  # Adjust position
                xanchor="center",
                x=0.5
            )
        )

        return fig, table_data

    else:
        raise ValueError("No valid portfolios found. Ensure your data has valid returns and volatility.")

if __name__ == "__main__":
    app.run(debug=True)
