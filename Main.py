from Main_Binance import run_binance
from Main_Interactive import run_interactive

Broker = "Binance"
# Broker = "Interactive Broker"

print(f"\nThe Broker is {Broker}\n")

if Broker == "Binance":
    run_binance()
elif Broker == "Interactive Broker":
    run_interactive()
