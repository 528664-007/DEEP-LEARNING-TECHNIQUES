# McCulloch-Pitts Neuron Implementation for Logical Functions

def mcCullochPittsNeuron(inputs, weights, threshold):
    """
    Implements a McCulloch-Pitts neuron.
    
    Parameters:
    inputs (list): List of binary input values (0 or 1).
    weights (list): Corresponding weights for each input.
    threshold (int): Activation threshold.

    Returns:
    int: Output (0 or 1).
    """
    net_input = sum(i * w for i, w in zip(inputs, weights))
    return 1 if net_input >= threshold else 0

# Implementing AND Function
def AND_gate(x1, x2):
    return mcCullochPittsNeuron([x1, x2], [1, 1], 2)

# Implementing OR Function
def OR_gate(x1, x2):
    return mcCullochPittsNeuron([x1, x2], [1, 1], 1)

# Implementing NOT Function
def NOT_gate(x):
    return mcCullochPittsNeuron([x], [-1], 0)

# Testing the logic functions
print("AND Gate:")
print(f"AND(0, 0) = {AND_gate(0, 0)}")
print(f"AND(0, 1) = {AND_gate(0, 1)}")
print(f"AND(1, 0) = {AND_gate(1, 0)}")
print(f"AND(1, 1) = {AND_gate(1, 1)}")

print("\nOR Gate:")
print(f"OR(0, 0) = {OR_gate(0, 0)}")
print(f"OR(0, 1) = {OR_gate(0, 1)}")
print(f"OR(1, 0) = {OR_gate(1, 0)}")
print(f"OR(1, 1) = {OR_gate(1, 1)}")

print("\nNOT Gate:")
print(f"NOT(0) = {NOT_gate(0)}")
print(f"NOT(1) = {NOT_gate(1)}")