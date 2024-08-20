
# price-tail-risk-sim

This project contains simulation of tail risk for financial securities.
Current focus on Gold CFD (XAUUSD).



For now the main method is Generative adversarial network(GAN) as main simulator.

[models](./models) contains main models and network architecture

For training you can use [gan-tail-price notebook](./gan-tail-price.ipynb) it is suggested to have comet installed and use you own API key for plotting the train.

Also for simulation you can see the [gan-tail-price-sim notebook](./gan-tail-price-sim.ipynb)