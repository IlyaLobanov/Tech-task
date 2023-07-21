import os
import random

def read_file(filename):

    with open(filename, 'r') as file:
        lines = file.readlines()

    orderbooks= []
    prices = []
    orderbook = []
    
    for line in lines:

        if "price:" in line:
            orderbooks.append(orderbook)
            action, price = line.split(":")
            prices.append((action.strip(), int(price.strip())))
            orderbook = []
        else: 
            parts = line.split("\t")
            if len(parts) == 3: 
                orderbook.append((int(parts[0]), int(parts[1]), parts[2].strip()))
    
    return orderbooks, prices



def split_data(glasses, prices, test_size=100):
    
    data = list(zip(glasses, prices))
    
    
    random.shuffle(data)
    
    
    train_data = data[test_size:]
    test_data = data[:test_size]
    
    
    train_glasses, train_prices = zip(*train_data)
    test_glasses, test_prices = zip(*test_data)
    
    return train_glasses, train_prices, test_glasses, test_prices

