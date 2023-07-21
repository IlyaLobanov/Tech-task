import matplotlib.pyplot as plt
import numpy as np

def calculate_statistics(glass, price):
    buy_prices = [entry[0] for entry in glass if entry[2] == 'Buy']
    sell_prices = [entry[0] for entry in glass if entry[2] == 'Sell']
        
    buy_volumes = [entry[1] for entry in glass if entry[2] == 'Buy']
    sell_volumes = [entry[1] for entry in glass if entry[2] == 'Sell']
        
    
    avg_buy_price = sum(buy_prices) / len(buy_prices) if buy_prices else 0
    avg_sell_price = sum(sell_prices) / len(sell_prices) if sell_prices else 0
        
    
    max_buy_price = max(buy_prices) if buy_prices else 0
    min_buy_price = min(buy_prices) if buy_prices else 0
        
    max_sell_price = max(sell_prices) if sell_prices else 0
    min_sell_price = min(sell_prices) if sell_prices else 0
        
    
    total_buy_volume = sum(buy_volumes)
    total_sell_volume = sum(sell_volumes)
        
    
    avg_buy_volume = total_buy_volume / len(buy_volumes) if buy_volumes else 0
    avg_sell_volume = total_sell_volume / len(sell_volumes) if sell_volumes else 0
        

    algorithm_price = price

    print(f"Average Buy Price: {avg_buy_price}, Average Sell Price: {avg_sell_price}")
    print(f"Max Buy Price: {max_buy_price}, Min Buy Price: {min_buy_price}")
    print(f"Max Sell Price: {max_sell_price}, Min Sell Price: {min_sell_price}")
    print(f"Total Buy Volume: {total_buy_volume}, Total Sell Volume: {total_sell_volume}")
    print(f"Average Buy Volume: {avg_buy_volume}, Average Sell Volume: {avg_sell_volume}")
    print(f'Algorithm price: {algorithm_price}')
    print('---')



def plot_glass(glass, algorithm_price):

    buys = [pair for pair in glass if pair[2] == 'Buy']
    sells = [pair for pair in glass if pair[2] == 'Sell']

    buys = sorted(buys, key=lambda x: x[0])
    sells = sorted(sells, key=lambda x: x[0])

    buy_prices = [pair[0] for pair in buys]
    buy_volumes = [pair[1] for pair in buys]
    sell_prices = [pair[0] for pair in sells]
    sell_volumes = [pair[1] for pair in sells]

    spread = sell_prices[0] - buy_prices[-1] if buys and sells else 0
    midprice = (sell_prices[0] + buy_prices[-1])/2 if buys and sells else 0

    direction = algorithm_price[0]
    plt.figure(figsize=(15, 20))
    plt.barh(buy_prices, buy_volumes, height=0.3, align='center', color='green', label='Buy')
    plt.barh(sell_prices, sell_volumes, height=0.3, align='center', color='red', label='Sell')

    plt.axhline(y=algorithm_price[1], color='blue', linestyle='--')
    plt.text(x = plt.xlim()[1], y = algorithm_price[1], s = f' Algorithm: {direction}', va = 'center')

    plt.axhline(y=midprice, color='purple', linestyle='--')
    plt.text(x = plt.xlim()[1], y = midprice, s = f' Mid Price', va = 'center')

    plt.legend(title=f'Spread: {spread}')
    plt.ylabel('Price')
    plt.xlabel('Volume')
    plt.title('Order Book')
    plt.grid(True)
    plt.yticks(list(plt.yticks()[0]) + [midprice, algorithm_price[1]])
    plt.show()


def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(a == b for a, b in zip(y_true, y_pred))
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def best_bid(lob):
    return max([price for price, volume, side in lob if side == "Buy"])

def best_ask(lob):
    return min([price for price, volume, side in lob if side == "Sell"])

def spread(lob):
    return best_ask(lob) - best_bid(lob)

def mid_price(lob):
    return (best_ask(lob) + best_bid(lob)) / 2

def vwap(lob):
    total_volume = sum(volume for price, volume, side in lob)
    total_value = sum(price * volume for price, volume, side in lob)
    return total_value / total_volume if total_volume != 0 else 0

def market_depth(order_book, side):
    return sum([price * size for price, size, s in order_book if s == side])


def weighted_midprice(order_book):
    total_sum = sum([price * size for price, size, side in order_book])
    total_size = sum([size for price, size, side in order_book])
    return total_sum / total_size if total_size > 0 else None

def price_range(order_book):
    max_price = max([price for price, size, side in order_book])
    min_price = min([price for price, size, side in order_book])
    return max_price - min_price

def relative_spread(order_book):
    best_bid = max([price for price, size, side in order_book if side == "Buy"])
    best_ask = min([price for price, size, side in order_book if side == "Sell"])
    return (best_ask - best_bid) / ((best_ask + best_bid) / 2)

def bid_ask_volume(order_book, side):
    return sum([size for price, size, s in order_book if s == side])

def best_bid_price(order_book):
    return max([price for price, size, side in order_book if side == "Buy"])

def best_ask_price(order_book):
    return min([price for price, size, side in order_book if side == "Sell"])

def avg_order_size(order_book, side):
    volumes = [size for price, size, s in order_book if s == side]
    return sum(volumes) / len(volumes) if volumes else 0

def std_order_size(order_book, side):
    volumes = [size for price, size, s in order_book if s == side]
    return np.std(volumes) if volumes else 0

def identify_level_and_direction(order_book, price):
    buy_arr = [order[0] for order in order_book if order[2] == 'Buy']
    sell_arr = [order[0] for order in order_book if order[2] == 'Sell']

    for i in range(len(buy_arr)):
        if price == buy_arr[i]:
            return f'bid_{i}'
    for i in range(-1, -len(sell_arr), -1):
        if price == sell_arr[i]:
            index = -i - 1
            return f'ask_{index}'
    
    return 'Price not found'

def check_gaps(ob, side):
    side_prices = [tp[0] for tp in ob if tp[2] == side]
    for p in range(1, len(side_prices)):
        if abs(side_prices[p] - side_prices[p - 1]) != 5:
            return False
    return True

def calculate_density(order_book):
    # Создаем словарь для хранения плотности на каждом уровне цены
    density_dict = {'Sell': {}, 'Buy': {}}
    cumulative_volume = {'Sell': 0, 'Buy': 0}
    max_price = {'Sell': max([price for price, volume, order_type in order_book if order_type == 'Sell']),
                 'Buy': min([price for price, volume, order_type in order_book if order_type == 'Buy'])}

    # Сортируем стакан по цене
    order_book.sort(key=lambda x: x[0])

    for price, volume, order_type in order_book:
        cumulative_volume[order_type] += volume

        if order_type == 'Sell':
            density = cumulative_volume[order_type] / (max_price[order_type] - price + 1)  # +1 чтобы избежать деления на 0
        else:  # Buy
            density = cumulative_volume[order_type] / (price - max_price[order_type] + 1)  # +1 чтобы избежать деления на 0

        density_dict[order_type][price] = density

    return density_dict



