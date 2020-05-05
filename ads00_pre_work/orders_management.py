"""
ORDER AND CART MANAGEMENT

Orders are sets of items ordered by a customer
An ordered item has four components:
 - a name
 - a quantity (the number of such items bought)
 - a unit price (in pence)
 - a unit weight (in pounds)
Those are represented by a tuple.

NOTE: You can safely assume that all orders have all the required fields
(name, quantity, unit-price and unit-weight) so no validation needs to be made.

DO NOT MODIFY CONSTANTS
"""

ORDER_SAMPLE_1 = {("lamp", 2, 2399, 2), ("chair", 4,
                                         3199, 10), ("table", 1, 5599, 85)}
ORDER_SAMPLE_2 = {("sofa", 1, 18399, 140), ("bookshelf", 2, 4799, 40)}

CATALOGUE = {("table", 9999, 20), ("chair", 2999, 5), ("lamp", 1999, 10)}


def delivery_charges(order):
    """
    Compute the delivery charges for an order. The company charges a flat £50
    fee plus £20 for each 100lbs (additional weight under 100lbs is ignored).

    E.g., delivery_charges({("desk", 1, 11999, 160)}) is 7000 (pence)
    E.g., delivery_charges({("desk", 2, 11999, 160)}) is 11000 (pence)
    E.g., delivery_charges({("lamp", 1, 2399, 2)}) is 5000 (pence)
    E.g., delivery_charges({("lamp", 50, 2399, 2)}) is 7000 (pence)

    :param order: order to process. See samples for examples.
    :return: delivery fee in pence
    :rtype: float | int
    """
    final_charges = 0
    weight = 0
    for item in order:
        weight += item[1]*item[3]

    final_charges = 100*(50 + 20*int(weight/100))

    return final_charges


def total_charge(order):
    """
    Compute the total charge for an order. It includes:
        - total price of items,
        - VAT (20% of the price of items),
        - delivery fee

    NOTE: in this computation, VAT is not applied to the delivery

    E.g., total_charge({("desk", 2, 11999, 160)}) is 39797 (pence)
    E.g., total_charge({("lamp", 50, 2399, 2)}) is 150940 (pence)

    Hint: Look up the built-in Python function round().

    :param order: order to process. See samples.
    :return: total price, in pence, rounded to the nearest penny.
    :rtype: float | int
    """
    item_price = 0
    for item in order:
        item_price += item[1]*item[2]

    final_charge = round(1.2*item_price + delivery_charges(order))
    return final_charge


print(total_charge({("desk", 2, 11999, 160)}))
print(total_charge({("lamp", 50, 2399, 2)}))


def add_item_to_order(name, quantity, order):
    """
    When a customer adds items to their basket, you need to update their order.

    The customer provides some of the details (the name of the item and
    the quantity they want). The rest (price and weight) needs to be looked up in
    the CATALOGUE provided above.

    NOTE: you must return a new order as a set and leave the argument unmodified.

    NOTE: if the order already contains some of the items, you must update the
    quantity field for that item; otherwise, you must add a new entry in the
    order

    NOTE: if the item cannot be found in the catalogue, the function should raise
    a KeyError.

    E.g., add_item_to_order("table", 1, {("table", 1, 9999, 20)}) is
        {("table", 2, 9999, 20)}
    E.g., add_item_to_order("chair", 1, {("table", 1, 9999, 20)}) is
        {("table", 1, 9999, 20), ("chair", 1, 2999, 5)}

    :param name: name of the item to add
    :param quantity: number of items to add
    :param order: previous order
    :return: a new order with the added items. If the item is unknown, raise a KeyError
    """
    # First check if they already have the order in their basket
    updated_order = set()
    in_order = False
    in_cat = False

    for item in order:
        if item[0] == name:
            in_order = True
            item_updated = (item[0], item[1] + quantity, item[2], item[3])
        else:
            item_updated = item

        updated_order.add(item_updated)

    if in_order is False:
        for item in CATALOGUE:
            if item[0] == name:
                in_cat = True
                item_updated = (item[0], quantity, item[1], item[2])

        if in_cat is False:
            raise KeyError

        updated_order.add(item_updated)
    return updated_order
