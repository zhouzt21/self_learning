
class Product:
    def __init__(self, name, price, discount):
        self.name = name
        self.price = price
        self.discount = discount

    def get_discount_price(self):
        return self.price * (1 - self.discount)

    def __str__(self):
        return f"Product: {self.name}, Price: {self.price}, Discount: {self.discount}, Discount Price: {self.get_discount_price()}"
    
class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, product, quantity):
        for _ in range(quantity):
            self.items.append(product)
    
    def remove_item(self, product):
        self.items.remove(product)
    
    def print_item_info(self, index):
        print(f"name:{self.items[index].name}")
        print(f"single price:{self.items[index].price}")
        print(f"single discount:{self.items[index].discount}")
        print(f"this type of items total price:{self.items[index].get_discount_price()}")

    def total_price(self):
        prices = [item.get_discount_price() for item in self.items]
        return sum(prices)

if __name__ == '__main__':
    p1 = Product("apple", 5, 0.1, )
    p2 = Product("banana", 3, 0.2)
    print(p1)
    print(p2)
    cart = ShoppingCart()
    cart.add_item(p1, 2)
    cart.add_item(p2, 3)
    for item in cart.items:
        print(item)