
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from dividend_optimizer import optimize_dividend_portfolio, get_market_data
    
    print("Testing get_market_data()...")
    df = get_market_data()
    if df is not None:
        print(f"✅ Market Data Loaded: {df.shape}")
        print(df.head())
    else:
        print("❌ Market Data Load Failed")

    print("\nTesting optimize_dividend_portfolio()...")
    result = optimize_dividend_portfolio(
        alpha=0.5,
        frequency='monthly',
        initial_investment=1000,
        universe_size=20 
    )
    
    if result:
        print("\n✅ Optimization Success")
        print(f"Keys: {result.keys()}")
        print(f"Portfolio: {len(result.get('portfolio', []))} items")
        if result.get('portfolio'):
            print(result['portfolio'][0])
        print(f"MVSK used: {result.get('_mvsk')}")
    else:
        print("❌ Optimization Returned None")

except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()
