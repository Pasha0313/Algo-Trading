from Main_Binance import run_binance 
from Main_Interactive import run_interactive

Mode = "backtest"   # other options:"optimize", "backtest", "live"

Broker = "Binance"
# Broker = "Interactive Broker"

print(f"\nThe Broker is {Broker}\n")
print(f"\nThe Mode is {Mode}\n")

if Broker == "Binance":
    run_binance(Mode=Mode)
elif Broker == "Interactive Broker":
    run_interactive()


