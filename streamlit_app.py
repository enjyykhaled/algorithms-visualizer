import heapq
import streamlit as st
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from heapq import heappop, heappush
import json
import sys
from collections import defaultdict, deque
import math

# Set page config
st.set_page_config(
    page_title="CS 205 Algorithms Visualizer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 10px;
    }
    h2 {
        color: #3498db;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .algorithm-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'array' not in st.session_state:
    st.session_state.array = []
if 'sorted_array' not in st.session_state:
    st.session_state.sorted_array = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}

# Performance tracking decorator
def track_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        comparisons = 0
        assignments = 0
        swaps = 0
        
        # For counting operations
        def count_comparison():
            nonlocal comparisons
            comparisons += 1
            return comparisons
        
        def count_assignment():
            nonlocal assignments
            assignments += 1
            return assignments
        
        def count_swap():
            nonlocal swaps
            swaps += 1
            return swaps
        
        kwargs['count_comparison'] = count_comparison
        kwargs['count_assignment'] = count_assignment
        kwargs['count_swap'] = count_swap
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # in milliseconds
        
        # Store metrics
        func_name = func.__name__.replace('_', ' ').title()
        st.session_state.performance_metrics[func_name] = {
            'comparisons': comparisons,
            'assignments': assignments,
            'swaps': swaps,
            'execution_time': execution_time,
            'space_complexity': sys.getsizeof(result) if result else 0
        }
        
        return result
    return wrapper

# Sorting Algorithms
@track_performance
def merge_sort(arr, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], count_comparison=count_comparison, count_assignment=count_assignment)
    right = merge_sort(arr[mid:], count_comparison=count_comparison, count_assignment=count_assignment)
    
    return merge(left, right, count_comparison, count_assignment)

def merge(left, right, count_comparison, count_assignment):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        count_comparison()
        if left[i] < right[j]:
            result.append(left[i])
            count_assignment()
            i += 1
        else:
            result.append(right[j])
            count_assignment()
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    count_assignment()
    count_assignment()
    
    return result

@track_performance
def quick_sort(arr, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_swap = kwargs.get('count_swap', lambda: None)
    
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    for _ in left + middle + right:
        count_comparison()
    
    return quick_sort(left, count_comparison=count_comparison, count_swap=count_swap) + \
           middle + \
           quick_sort(right, count_comparison=count_comparison, count_swap=count_swap)

@track_performance
def bubble_sort(arr, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_swap = kwargs.get('count_swap', lambda: None)
    
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            count_comparison()
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                count_swap()
    return arr

@track_performance
def selection_sort(arr, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_swap = kwargs.get('count_swap', lambda: None)
    
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            count_comparison()
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        count_swap()
    return arr

@track_performance
def insertion_sort(arr, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_swap = kwargs.get('count_swap', lambda: None)
    
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            count_comparison()
            arr[j+1] = arr[j]
            count_swap()
            j -= 1
        arr[j+1] = key
        count_swap()
    return arr

@track_performance
def heap_sort(arr, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_swap = kwargs.get('count_swap', lambda: None)
    
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n:
            count_comparison()
            if arr[left] > arr[largest]:
                largest = left
        
        if right < n:
            count_comparison()
            if arr[right] > arr[largest]:
                largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            count_swap()
            heapify(arr, n, largest)
    
    n = len(arr)
    
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        count_swap()
        heapify(arr, i, 0)
    
    return arr

# Dynamic Programming Algorithms
@track_performance
def fibonacci_memo(n, memo=None, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    if memo is None:
        memo = {}
    
    count_comparison()
    if n in memo:
        return memo[n]
    
    count_comparison()
    if n <= 2:
        return 1
    
    memo[n] = fibonacci_memo(n-1, memo, count_comparison=count_comparison, count_assignment=count_assignment) + \
              fibonacci_memo(n-2, memo, count_comparison=count_comparison, count_assignment=count_assignment)
    count_assignment()
    
    return memo[n]

@track_performance
def fibonacci_tab(n, **kwargs):
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    if n == 0:
        return 0
    
    table = [0] * (n + 1)
    table[1] = 1
    count_assignment()
    count_assignment()
    
    for i in range(2, n + 1):
        table[i] = table[i-1] + table[i-2]
        count_assignment()
    
    return table[n]

@track_performance
def knapsack_01(values, weights, capacity, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    count_assignment()
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            count_comparison()
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                count_assignment()
            else:
                dp[i][w] = dp[i-1][w]
                count_assignment()
    
    return dp[n][capacity]

@track_performance
def knapsack_fractional(values, weights, capacity, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    items = list(zip(values, weights))
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    count_assignment()
    
    total_value = 0.0
    remaining_capacity = capacity
    
    for value, weight in items:
        count_comparison()
        if remaining_capacity <= 0:
            break
        
        count_comparison()
        if weight <= remaining_capacity:
            total_value += value
            remaining_capacity -= weight
            count_assignment()
            count_assignment()
        else:
            fraction = remaining_capacity / weight
            total_value += value * fraction
            remaining_capacity = 0
            count_assignment()
            count_assignment()
    
    return total_value

@track_performance
def longest_common_subsequence(text1, text2, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    count_assignment()
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            count_comparison()
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                count_assignment()
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                count_assignment()
    
    return dp[m][n]

@track_performance
def floyd_warshall(graph, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    n = len(graph)
    dist = [[0] * n for _ in range(n)]
    count_assignment()
    
    for i in range(n):
        for j in range(n):
            dist[i][j] = graph[i][j]
            count_assignment()
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                count_comparison()
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    count_assignment()
    
    return dist

# Greedy Algorithms
@track_performance
def coin_change_greedy(coins, amount, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    coins.sort(reverse=True)
    count_assignment()
    
    result = []
    remaining = amount
    
    for coin in coins:
        count_comparison()
        while remaining >= coin:
            result.append(coin)
            remaining -= coin
            count_assignment()
            count_assignment()
    
    return result if remaining == 0 else []

@track_performance
def activity_selection(start, end, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    activities = list(zip(start, end))
    activities.sort(key=lambda x: x[1])
    count_assignment()
    
    selected = []
    last_end = -1
    
    for s, e in activities:
        count_comparison()
        if s >= last_end:
            selected.append((s, e))
            last_end = e
            count_assignment()
            count_assignment()
    
    return selected

@track_performance
def huffman_coding(text, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    frequency = defaultdict(int)
    for char in text:
        frequency[char] += 1
        count_assignment()
    
    heap = [[weight, [char, ""]] for char, weight in frequency.items()]
    count_assignment()
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        count_assignment()
        count_assignment()
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
            count_assignment()
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
            count_assignment()
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        count_assignment()
    
    huffman_dict = {}
    for pair in heap[0][1:]:
        huffman_dict[pair[0]] = pair[1]
        count_assignment()
    
    return huffman_dict

@track_performance
def kruskal_mst(graph, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    parent = {}
    rank = {}
    
    def find(node):
        nonlocal count_assignment
        if parent[node] != node:
            parent[node] = find(parent[node])
            count_assignment()
        return parent[node]
    
    def union(node1, node2):
        nonlocal count_comparison, count_assignment
        root1 = find(node1)
        root2 = find(node2)
        
        count_comparison()
        if root1 != root2:
            count_comparison()
            if rank[root1] > rank[root2]:
                parent[root2] = root1
                count_assignment()
            else:
                parent[root1] = root2
                count_assignment()
                count_comparison()
                if rank[root1] == rank[root2]:
                    rank[root2] += 1
                    count_assignment()
    
    edges = []
    for u in graph:
        for v, weight in graph[u].items():
            edges.append((weight, u, v))
            count_assignment()
    
    edges.sort()
    count_assignment()
    
    for node in graph:
        parent[node] = node
        rank[node] = 0
        count_assignment()
        count_assignment()
    
    mst = []
    for edge in edges:
        weight, u, v = edge
        count_comparison()
        if find(u) != find(v):
            union(u, v)
            mst.append(edge)
            count_assignment()
    
    return mst

# Binary Search
@track_performance
def binary_search(arr, target, **kwargs):
    count_comparison = kwargs.get('count_comparison', lambda: None)
    count_assignment = kwargs.get('count_assignment', lambda: None)
    
    low = 0
    high = len(arr) - 1
    steps = []
    
    while low <= high:
        mid = (low + high) // 2
        steps.append(mid)
        count_assignment()
        
        count_comparison()
        if arr[mid] == target:
            return (True, mid, steps)
        
        count_comparison()
        if arr[mid] < target:
            low = mid + 1
            count_assignment()
        else:
            high = mid - 1
            count_assignment()
    
    return (False, -1, steps)

# Helper functions
def generate_array(size, array_type):
    if array_type == "Sorted":
        return list(range(1, size + 1))
    elif array_type == "Inversely Sorted":
        return list(range(size, 0, -1))
    elif array_type == "Random":
        return random.sample(range(1, size + 1), size)
    elif array_type == "Custom":
        custom_input = st.text_input("Enter comma-separated numbers:")
        if custom_input:
            return [int(x.strip()) for x in custom_input.split(",")]
        return []
    return []

def create_graph(num_vertices, edge_probability=0.3, max_weight=10):
    graph = [[float('inf')] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        graph[i][i] = 0
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                weight = random.randint(1, max_weight)
                graph[i][j] = weight
                graph[j][i] = weight
    return graph

def create_weighted_graph(num_nodes, edge_probability=0.4, max_weight=20):
    graph = defaultdict(dict)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                weight = random.randint(1, max_weight)
                graph[str(i)][str(j)] = weight
                graph[str(j)][str(i)] = weight
    return graph

def display_sorting_visualization(arr, sorted_arr, steps=None, target=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original array
    ax1.bar(range(len(arr)), arr, color='skyblue')
    ax1.set_title('Original Array')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    
    # Sorted array
    colors = ['lightgreen'] * len(sorted_arr)
    if target is not None and steps:
        for step in steps:
            if step < len(colors):
                colors[step] = 'red'
    ax2.bar(range(len(sorted_arr)), sorted_arr, color=colors)
    ax2.set_title('Sorted Array')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    
    st.pyplot(fig)

def display_binary_search_visualization(arr, steps, target):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(arr)), arr, color='lightblue')
    
    for i, step in enumerate(steps):
        if arr[step] == target:
            ax.bar(step, arr[step], color='green')
            break
        else:
            ax.bar(step, arr[step], color='red')
    
    ax.set_title('Binary Search Steps (Red: Checks, Green: Found)')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    st.pyplot(fig)

def display_performance_metrics(metrics):
    if not metrics:
        st.warning("No performance metrics available yet.")
        return
    
    st.subheader("Performance Metrics Comparison")
    
    algorithms = list(metrics.keys())
    comparisons = [metrics[alg]['comparisons'] for alg in algorithms]
    assignments = [metrics[alg]['assignments'] for alg in algorithms]
    swaps = [metrics[alg]['swaps'] for alg in algorithms if 'swaps' in metrics[alg]]
    times = [metrics[alg]['execution_time'] for alg in algorithms]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Comparisons
    axes[0, 0].bar(algorithms, comparisons, color='skyblue')
    axes[0, 0].set_title('Number of Comparisons')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Assignments
    axes[0, 1].bar(algorithms, assignments, color='lightgreen')
    axes[0, 1].set_title('Number of Assignments')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Swaps (if applicable)
    if swaps:
        axes[1, 0].bar(algorithms[:len(swaps)], swaps, color='salmon')
        axes[1, 0].set_title('Number of Swaps')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].axis('off')
    
    # Execution Time
    axes[1, 1].bar(algorithms, times, color='gold')
    axes[1, 1].set_title('Execution Time (ms)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Metrics table
    st.subheader("Detailed Metrics")
    metrics_data = []
    for alg, data in metrics.items():
        row = {
            'Algorithm': alg,
            'Comparisons': data['comparisons'],
            'Assignments': data['assignments'],
            'Execution Time (ms)': f"{data['execution_time']:.4f}",
            'Space (bytes)': data['space_complexity']
        }
        if 'swaps' in data:
            row['Swaps'] = data['swaps']
        metrics_data.append(row)
    
    st.table(pd.DataFrame(metrics_data))

# Main App
def main():
    st.title("CS 205 Algorithms Design Project")
    st.markdown("""
    An interactive GUI-based application to demonstrate and compare different algorithms including:
    - Sorting algorithms
    - Dynamic programming techniques
    - Greedy algorithms
    - Binary search
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        algorithm_category = st.selectbox(
            "Select Algorithm Category",
            ["Sorting Algorithms", "Dynamic Programming", "Greedy Algorithms", "Binary Search"]
        )
        
        if algorithm_category == "Sorting Algorithms":
            array_type = st.selectbox(
                "Array Type",
                ["Sorted", "Inversely Sorted", "Random", "Custom"]
            )
            array_size = st.slider("Array Size", 5, 100, 20)
            
            sorting_algorithm = st.selectbox(
                "Sorting Algorithm",
                ["Merge Sort", "Quick Sort", "Bubble Sort", "Selection Sort", "Insertion Sort", "Heap Sort"]
            )
            
            if st.button("Generate and Sort"):
                st.session_state.array = generate_array(array_size, array_type)
                if st.session_state.array:
                    if sorting_algorithm == "Merge Sort":
                        st.session_state.sorted_array = merge_sort(st.session_state.array.copy())
                    elif sorting_algorithm == "Quick Sort":
                        st.session_state.sorted_array = quick_sort(st.session_state.array.copy())
                    elif sorting_algorithm == "Bubble Sort":
                        st.session_state.sorted_array = bubble_sort(st.session_state.array.copy())
                    elif sorting_algorithm == "Selection Sort":
                        st.session_state.sorted_array = selection_sort(st.session_state.array.copy())
                    elif sorting_algorithm == "Insertion Sort":
                        st.session_state.sorted_array = insertion_sort(st.session_state.array.copy())
                    elif sorting_algorithm == "Heap Sort":
                        st.session_state.sorted_array = heap_sort(st.session_state.array.copy())
        
        elif algorithm_category == "Dynamic Programming":
            dp_problem = st.selectbox(
                "Select Problem",
                ["Fibonacci Sequence", "Knapsack Problem", "Longest Common Subsequence", "Shortest Path (Floyd-Warshall)"]
            )
            
            if dp_problem == "Fibonacci Sequence":
                fib_n = st.slider("Fibonacci term (n)", 1, 40, 10)
                if st.button("Calculate Fibonacci"):
                    st.session_state.fib_memo = fibonacci_memo(fib_n)
                    st.session_state.fib_tab = fibonacci_tab(fib_n)
            
            elif dp_problem == "Knapsack Problem":
                knapsack_type = st.radio("Knapsack Type", ["0/1 Knapsack", "Fractional Knapsack"])
                capacity = st.slider("Capacity", 1, 100, 50)
                num_items = st.slider("Number of Items", 1, 20, 5)
                
                if st.button("Generate and Solve"):
                    values = [random.randint(10, 100) for _ in range(num_items)]
                    weights = [random.randint(1, 30) for _ in range(num_items)]
                    
                    if knapsack_type == "0/1 Knapsack":
                        st.session_state.knapsack_result = knapsack_01(values, weights, capacity)
                    else:
                        st.session_state.knapsack_result = knapsack_fractional(values, weights, capacity)
                    
                    st.session_state.knapsack_values = values
                    st.session_state.knapsack_weights = weights
            
            elif dp_problem == "Longest Common Subsequence":
                lcs_text1 = st.text_input("First sequence", "AGGTAB")
                lcs_text2 = st.text_input("Second sequence", "GXTXAYB")
                if st.button("Find LCS"):
                    st.session_state.lcs_result = longest_common_subsequence(lcs_text1, lcs_text2)
            
            elif dp_problem == "Shortest Path (Floyd-Warshall)":
                num_vertices = st.slider("Number of vertices", 3, 10, 5)
                if st.button("Generate Graph and Find Shortest Paths"):
                    graph = create_graph(num_vertices)
                    st.session_state.graph = graph
                    st.session_state.shortest_paths = floyd_warshall(graph)
        
        elif algorithm_category == "Greedy Algorithms":
            greedy_problem = st.selectbox(
                "Select Problem",
                ["Coin Change", "Activity Selection", "Huffman Coding", "Kruskal's MST"]
            )
            
            if greedy_problem == "Coin Change":
                coins_input = st.text_input("Coin denominations (comma separated)", "25,10,5,1")
                amount = st.slider("Amount to make", 1, 100, 37)
                if st.button("Solve Coin Change"):
                    coins = [int(c.strip()) for c in coins_input.split(",")]
                    st.session_state.coin_change_result = coin_change_greedy(coins, amount)
            
            elif greedy_problem == "Activity Selection":
                num_activities = st.slider("Number of activities", 3, 15, 5)
                if st.button("Generate and Solve"):
                    start_times = sorted([random.randint(0, 20) for _ in range(num_activities)])
                    durations = [random.randint(1, 5) for _ in range(num_activities)]
                    end_times = [start + dur for start, dur in zip(start_times, durations)]
                    st.session_state.activity_start = start_times
                    st.session_state.activity_end = end_times
                    st.session_state.activity_result = activity_selection(start_times, end_times)
            
            elif greedy_problem == "Huffman Coding":
                huffman_text = st.text_input("Input text for Huffman coding", "ABBCCCDDDDEEEEE")
                if st.button("Generate Huffman Codes"):
                    st.session_state.huffman_result = huffman_coding(huffman_text)
            
            elif greedy_problem == "Kruskal's MST":
                num_nodes = st.slider("Number of nodes", 3, 10, 5)
                if st.button("Generate Graph and Find MST"):
                    graph = create_weighted_graph(num_nodes)
                    st.session_state.kruskal_graph = graph
                    st.session_state.mst_result = kruskal_mst(graph)
        
        elif algorithm_category == "Binary Search":
            array_type = st.selectbox(
                "Array Type",
                ["Sorted", "Inversely Sorted", "Random", "Custom"],
                key="binary_search_array_type"
            )
            array_size = st.slider("Array Size", 5, 100, 20, key="binary_search_array_size")
            target = st.number_input("Target value to search", min_value=1, max_value=array_size, value=10)
            
            if st.button("Generate and Search"):
                st.session_state.binary_search_array = generate_array(array_size, array_type)
                if st.session_state.binary_search_array:
                    st.session_state.binary_search_sorted = sorted(st.session_state.binary_search_array)
                    found, index, steps = binary_search(st.session_state.binary_search_sorted, target)
                    st.session_state.binary_search_result = (found, index, steps)
    
    # Main content area
    if algorithm_category == "Sorting Algorithms":
        st.header("Sorting Algorithms Visualization")
        
        if st.session_state.array:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Array")
                st.write(st.session_state.array)
            
            with col2:
                st.subheader("Sorted Array")
                st.write(st.session_state.sorted_array)
            
            display_sorting_visualization(st.session_state.array, st.session_state.sorted_array)
            
            if sorting_algorithm in st.session_state.performance_metrics:
                metrics = st.session_state.performance_metrics[sorting_algorithm]
                
                st.subheader("Performance Metrics")
                cols = st.columns(4)
                cols[0].metric("Comparisons", metrics['comparisons'])
                cols[1].metric("Assignments", metrics['assignments'])
                cols[2].metric("Swaps", metrics.get('swaps', 'N/A'))
                cols[3].metric("Time (ms)", f"{metrics['execution_time']:.4f}")
        
        else:
            st.info("Generate an array and select a sorting algorithm to visualize.")
    
    elif algorithm_category == "Dynamic Programming":
        st.header("Dynamic Programming Algorithms")
        
        if dp_problem == "Fibonacci Sequence":
            st.subheader("Fibonacci Sequence")
            st.markdown("""
            Compare memoization (top-down) vs. tabulation (bottom-up) approaches.
            """)
            
            if 'fib_memo' in st.session_state:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Memoization (Top-Down)")
                    st.write(f"Fibonacci({fib_n}) = {st.session_state.fib_memo}")
                    if 'Fibonacci Memo' in st.session_state.performance_metrics:
                        metrics = st.session_state.performance_metrics['Fibonacci Memo']
                        st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                        st.metric("Space (bytes)", metrics['space_complexity'])
                
                with col2:
                    st.markdown("### Tabulation (Bottom-Up)")
                    st.write(f"Fibonacci({fib_n}) = {st.session_state.fib_tab}")
                    if 'Fibonacci Tab' in st.session_state.performance_metrics:
                        metrics = st.session_state.performance_metrics['Fibonacci Tab']
                        st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                        st.metric("Space (bytes)", metrics['space_complexity'])
            
            else:
                st.info("Select a Fibonacci term and click 'Calculate Fibonacci'")
        
        elif dp_problem == "Knapsack Problem":
            st.subheader("Knapsack Problem")
            
            if 'knapsack_result' in st.session_state:
                st.write(f"### {knapsack_type} Solution")
                st.write(f"Maximum value: {st.session_state.knapsack_result}")
                
                st.write("### Items:")
                items_df = pd.DataFrame({
                    'Value': st.session_state.knapsack_values,
                    'Weight': st.session_state.knapsack_weights
                })
                st.table(items_df)
                
                if knapsack_type == "0/1 Knapsack":
                    if 'Knapsack 01' in st.session_state.performance_metrics:
                        metrics = st.session_state.performance_metrics['Knapsack 01']
                        st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                        st.metric("Space (bytes)", metrics['space_complexity'])
                else:
                    if 'Knapsack Fractional' in st.session_state.performance_metrics:
                        metrics = st.session_state.performance_metrics['Knapsack Fractional']
                        st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                        st.metric("Space (bytes)", metrics['space_complexity'])
            
            else:
                st.info("Configure the knapsack problem and click 'Generate and Solve'")
        
        elif dp_problem == "Longest Common Subsequence":
            st.subheader("Longest Common Subsequence")
            
            if 'lcs_result' in st.session_state:
                st.write(f"Length of LCS: {st.session_state.lcs_result}")
                
                if 'Longest Common Subsequence' in st.session_state.performance_metrics:
                    metrics = st.session_state.performance_metrics['Longest Common Subsequence']
                    st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                    st.metric("Space (bytes)", metrics['space_complexity'])
            
            else:
                st.info("Enter two sequences and click 'Find LCS'")
        
        elif dp_problem == "Shortest Path (Floyd-Warshall)":
            st.subheader("Floyd-Warshall Algorithm")
            
            if 'shortest_paths' in st.session_state:
                st.write("### Generated Graph (Adjacency Matrix)")
                st.write(pd.DataFrame(st.session_state.graph))
                
                st.write("### Shortest Paths Between All Pairs")
                st.write(pd.DataFrame(st.session_state.shortest_paths))
                
                if 'Floyd Warshall' in st.session_state.performance_metrics:
                    metrics = st.session_state.performance_metrics['Floyd Warshall']
                    st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                    st.metric("Space (bytes)", metrics['space_complexity'])
            
            else:
                st.info("Generate a graph and click 'Generate Graph and Find Shortest Paths'")
    
    elif algorithm_category == "Greedy Algorithms":
        st.header("Greedy Algorithms")
        
        if greedy_problem == "Coin Change":
            st.subheader("Coin Change Problem")
            
            if 'coin_change_result' in st.session_state:
                st.write(f"Minimum coins needed: {len(st.session_state.coin_change_result)}")
                st.write(f"Coins used: {st.session_state.coin_change_result}")
                
                if 'Coin Change Greedy' in st.session_state.performance_metrics:
                    metrics = st.session_state.performance_metrics['Coin Change Greedy']
                    st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                    st.metric("Space (bytes)", metrics['space_complexity'])
            
            else:
                st.info("Enter coin denominations and amount, then click 'Solve Coin Change'")
        
        elif greedy_problem == "Activity Selection":
            st.subheader("Activity Selection Problem")
            
            if 'activity_result' in st.session_state:
                st.write(f"Maximum number of activities: {len(st.session_state.activity_result)}")
                st.write("Selected activities (start, end):")
                
                activities_df = pd.DataFrame({
                    'Start': st.session_state.activity_start,
                    'End': st.session_state.activity_end
                })
                st.table(activities_df)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 4))
                for i, (s, e) in enumerate(zip(st.session_state.activity_start, st.session_state.activity_end)):
                    color = 'green' if (s, e) in st.session_state.activity_result else 'red'
                    ax.barh(i, e-s, left=s, color=color, alpha=0.6)
                
                ax.set_yticks(range(len(st.session_state.activity_start)))
                ax.set_yticklabels([f"Activity {i+1}" for i in range(len(st.session_state.activity_start))])
                ax.set_xlabel('Time')
                ax.set_title('Activity Selection (Green: Selected, Red: Not Selected)')
                st.pyplot(fig)
                
                if 'Activity Selection' in st.session_state.performance_metrics:
                    metrics = st.session_state.performance_metrics['Activity Selection']
                    st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                    st.metric("Space (bytes)", metrics['space_complexity'])
            
            else:
                st.info("Generate activities and click 'Generate and Solve'")
        
        elif greedy_problem == "Huffman Coding":
            st.subheader("Huffman Coding")
            
            if 'huffman_result' in st.session_state:
                st.write("### Huffman Codes:")
                codes_df = pd.DataFrame.from_dict(st.session_state.huffman_result, orient='index', columns=['Code'])
                st.table(codes_df)
                
                # Visualization
                st.write("### Code Lengths:")
                code_lengths = {char: len(code) for char, code in st.session_state.huffman_result.items()}
                fig, ax = plt.subplots()
                ax.bar(code_lengths.keys(), code_lengths.values())
                ax.set_xlabel('Character')
                ax.set_ylabel('Code Length')
                st.pyplot(fig)
                
                if 'Huffman Coding' in st.session_state.performance_metrics:
                    metrics = st.session_state.performance_metrics['Huffman Coding']
                    st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                    st.metric("Space (bytes)", metrics['space_complexity'])
            
            else:
                st.info("Enter text and click 'Generate Huffman Codes'")
        
        elif greedy_problem == "Kruskal's MST":
            st.subheader("Kruskal's Minimum Spanning Tree")
            
            if 'mst_result' in st.session_state:
                st.write("### Original Graph:")
                st.json(st.session_state.kruskal_graph)
                
                st.write("### Minimum Spanning Tree Edges:")
                mst_edges = sorted([(u, v, w) for w, u, v in st.session_state.mst_result])
                st.table(pd.DataFrame(mst_edges, columns=['Node 1', 'Node 2', 'Weight']))
                
                total_weight = sum(w for w, _, _ in st.session_state.mst_result)
                st.write(f"Total MST Weight: {total_weight}")
                
                if 'Kruskal Mst' in st.session_state.performance_metrics:
                    metrics = st.session_state.performance_metrics['Kruskal Mst']
                    st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
                    st.metric("Space (bytes)", metrics['space_complexity'])
            
            else:
                st.info("Generate a graph and click 'Generate Graph and Find MST'")
    
    elif algorithm_category == "Binary Search":
        st.header("Binary Search Visualization")
        
        if 'binary_search_result' in st.session_state:
            found, index, steps = st.session_state.binary_search_result
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Array")
                st.write(st.session_state.binary_search_array)
            
            with col2:
                st.subheader("Sorted Array")
                st.write(st.session_state.binary_search_sorted)
            
            if found:
                st.success(f"Target {target} found at index {index}")
            else:
                st.error(f"Target {target} not found in the array")
            
            display_binary_search_visualization(st.session_state.binary_search_sorted, steps, target)
            
            if 'Binary Search' in st.session_state.performance_metrics:
                metrics = st.session_state.performance_metrics['Binary Search']
                st.metric("Comparisons", metrics['comparisons'])
                st.metric("Time (ms)", f"{metrics['execution_time']:.4f}")
        
        else:
            st.info("Generate an array and enter a target value to search.")
    
    # Performance comparison section
    if st.session_state.performance_metrics:
        st.header("Algorithm Performance Comparison")
        display_performance_metrics(st.session_state.performance_metrics)
    
    # Data import/export
    st.sidebar.header("Data Handling")
    if st.sidebar.button("Clear All Data"):
        st.session_state.clear()
        st.rerun()
    
    if st.sidebar.button("Export Performance Data"):
        if st.session_state.performance_metrics:
            metrics_df = pd.DataFrame.from_dict(st.session_state.performance_metrics, orient='index')
            csv = metrics_df.to_csv(index=True)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name="algorithm_performance.csv",
                mime="text/csv"
            )
        else:
            st.sidebar.warning("No performance data to export")

if __name__ == "__main__":
    main()