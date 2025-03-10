import streamlit as st
import simpy
import random
import pandas as pd
import plotly.express as px
import numpy as np

# --- Simulation Code --- (Corrected and Improved, Balking Removed)
class Config:
    def __init__(self, arrival_rate, order_time, prep_time, payment_time, order_queue_capacity, service_queue_capacity, simulation_time, num_order_stations):
        self.ARRIVAL_RATE = arrival_rate
        self.ORDER_TIME = order_time
        self.PREP_TIME = prep_time
        self.PAYMENT_TIME = payment_time
        self.ORDER_QUEUE_CAPACITY = order_queue_capacity  # Capacity for the queue *before* ordering
        self.SERVICE_QUEUE_CAPACITY = service_queue_capacity # Capacity for queue between order and payment
        self.SIMULATION_TIME = simulation_time
        self.NUM_ORDER_STATIONS = num_order_stations

class DriveThrough:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.order_stations = [simpy.Resource(env, capacity=1) for _ in range(config.NUM_ORDER_STATIONS)]
        self.payment_window = simpy.Resource(env, capacity=1)
        self.order_queue = simpy.Store(env, capacity=config.ORDER_QUEUE_CAPACITY)
        self.service_queue = simpy.Store(env, capacity=config.SERVICE_QUEUE_CAPACITY)
        self.order_prep = simpy.Resource(env, capacity=1)
        self.order_ready_events = {}
        self.metrics = {
            'wait_times_ordering': [],
            'wait_times_payment': [],
            'wait_times_before_order_queue': [],
            'wait_times_before_service': [],
            'total_times': [],
            'cars_served': 0,
            'cars_blocked_order_queue': 0,  # Still track blocking, but no balking logic
            'cars_blocked_service_queue': 0,
            'car_ids': [],
            'order_sizes': [],
            # 'balking_events': [],  Removed balking events
        }
        self.menu = {
            "Burger": [2.5, 0.3],
            "Fries": [1.5, 0.4],
            "Salad": [2.0, 0.15],
            "Drink": [0.5, 0.8],
            "Ice Cream": [1.0, 0.35],
        }
        self.payment_types = {
            "Card": [0.7, 0.85],
            "Cash": [1.1, 0.13],
            "MobilePay": [0.5, 0.02]
        }

    def process_car(self, car_id):
        print(f"Car {car_id} arrived at {self.env.now}")
        arrival_time = self.env.now
        self.metrics['car_ids'].append(car_id)

        # Initialize metrics with NaN
        for metric in ['wait_times_ordering', 'wait_times_payment', 'wait_times_before_order_queue',
                       'wait_times_before_service', 'total_times', 'order_sizes']:
            self.metrics[metric].append(np.nan)
        # self.metrics['balking_events'].append(0)  Removed balking initialization


        # --- Stage 1: Order Queue ---
        enter_order_queue_time = self.env.now
        try:
            yield self.order_queue.put(car_id)
            self.metrics['wait_times_before_order_queue'][-1] = self.env.now - enter_order_queue_time
            print(f"Car {car_id} entered order queue at {self.env.now}")

            # --- Stage 2: Ordering ---
            order_station = random.choice(self.order_stations)
            with order_station.request() as request:
                yield request
                yield self.order_queue.get()
                print(f"Car {car_id} left order queue at {self.env.now}")
                order_start_time = self.env.now

                order_size = random.randint(1, 4)
                self.metrics['order_sizes'][-1] = order_size
                order = [random.choices(list(self.menu.keys()), weights=[item[1] for item in self.menu.values()])[0]
                         for _ in range(order_size)]

                order_time = self.config.ORDER_TIME * (0.8 + 0.4 * order_size) * random.uniform(0.9, 1.1)
                yield self.env.timeout(order_time)
                self.metrics['wait_times_ordering'][-1] = self.env.now - order_start_time
                print(f"Car {car_id} finished ordering at {self.env.now}, Order: {order}")

            self.env.process(self.prep_order(car_id, order))

        except simpy.Interrupt:  # Keep the blocking logic
            print(f"Car {car_id} blocked at order queue at {self.env.now}")
            self.metrics['cars_blocked_order_queue'] += 1
            return

        # --- Stage 4: Service Queue ---
        enter_service_queue_time = self.env.now
        try:
            yield self.service_queue.put(car_id)
            self.metrics['wait_times_before_service'][-1] = self.env.now - enter_service_queue_time
            print(f"Car {car_id} entered service queue at {self.env.now}")

            # --- Stage 5: Payment and Handoff ---
            with self.payment_window.request() as request:
                yield request
                yield self.service_queue.get()
                print(f"Car {car_id} left service queue at {self.env.now}")
                service_start_time = self.env.now

                payment_type = random.choices(list(self.payment_types.keys()), weights=[pt[1] for pt in self.payment_types.values()])[0]
                payment_time = np.random.normal(loc=self.payment_types[payment_type][0], scale=self.payment_types[payment_type][0] / 4)
                payment_time = max(payment_time, self.payment_types[payment_type][0] / 4)  # Ensure non-negative
                if payment_type == "Card" and random.random() < 0.05:
                    payment_time += 1.5

                yield self.env.timeout(payment_time)
                self.metrics['wait_times_payment'][-1] = self.env.now - service_start_time
                print(f"Car {car_id} finished payment at {self.env.now}")

        except simpy.Interrupt:  # Keep the blocking logic
            print(f"Car {car_id} blocked at service queue at {self.env.now}")
            self.metrics['cars_blocked_service_queue'] += 1
            return

        # --- Stage 6: Wait for order prep ---
        yield self.order_ready_events[car_id]
        del self.order_ready_events[car_id]
        print(f"Car {car_id} order ready at {self.env.now}")

        # --- Completion ---
        self.metrics['total_times'][-1] = self.env.now - arrival_time
        self.metrics['cars_served'] += 1
        print(f"Car {car_id} completed at {self.env.now}")
    
    def prep_order(self, car_id, order):
        with self.order_prep.request() as req:
            yield req
            total_prep_time = 0
            for item in order:
                total_prep_time += self.menu[item][0]
            total_prep_time *= random.uniform(0.8, 1.2)
            yield self.env.timeout(total_prep_time)
            self.order_ready_events[car_id].succeed()

def car_arrivals(env, drive_through):
    car_id = 0
    while True:
        yield env.timeout(random.expovariate(drive_through.config.ARRIVAL_RATE))
        car_id += 1
        drive_through.order_ready_events[car_id] = env.event()
        env.process(drive_through.process_car(car_id))

def run_simulation(config):
    print("Starting simulation...")
    env = simpy.Environment()
    drive_through = DriveThrough(env, config)
    env.process(car_arrivals(env, drive_through))
    print("Car arrivals process started...")
    env.run(until=config.SIMULATION_TIME)
    print("Simulation completed.")
    return drive_through.metrics

def analyze_results(metrics, config):
    if not metrics['car_ids']:
        return {
            'Cars Served': 0,
            'Cars Blocked (Order Queue)': 0,
            'Cars Blocked (Service Queue)': 0,
            'Throughput (cars/hour)': 0.0,
            'Avg Wait Ordering (min)': 0.0,
            'Avg Wait Payment (min)': 0.0,
            'Avg Wait Before Order Queue (min)': 0.0,
            'Avg Wait Before Service (min)': 0.0,
            'Avg Total Time (min)': 0.0,
        }, px.histogram(), px.histogram(),pd.DataFrame()

    df = pd.DataFrame({
        'Car ID': metrics['car_ids'],
        'Wait Time Ordering (min)': metrics['wait_times_ordering'],
        'Wait Time Payment (min)': metrics['wait_times_payment'],
        'Wait Time Before Order Queue (min)': metrics['wait_times_before_order_queue'],
        'Wait Time Before Service (min)': metrics['wait_times_before_service'],
        'Total Time (min)': metrics['total_times']
    })

    avg_wait_ordering = df['Wait Time Ordering (min)'].mean()
    avg_wait_payment = df['Wait Time Payment (min)'].mean()
    avg_wait_before_order_queue = df['Wait Time Before Order Queue (min)'].mean()
    avg_wait_before_service = df['Wait Time Before Service (min)'].mean()
    avg_total_time = df['Total Time (min)'].mean()
    throughput = metrics['cars_served'] / config.SIMULATION_TIME * 60

    results = {
        'Cars Served': metrics['cars_served'],
        'Cars Blocked (Order Queue)': metrics['cars_blocked_order_queue'],
        'Cars Blocked (Service Queue)': metrics['cars_blocked_service_queue'],
        'Throughput (cars/hour)': f"{throughput:.2f}",
        'Avg Wait Ordering (min)': f"{avg_wait_ordering:.2f}",
        'Avg Wait Payment (min)': f"{avg_wait_payment:.2f}",
        'Avg Wait Before Order Queue (min)': f"{avg_wait_before_order_queue:.2f}",
        'Avg Wait Before Service (min)': f"{avg_wait_before_service:.2f}",
        'Avg Total Time (min)': f"{avg_total_time:.2f}",
    }

    fig_wait_order = px.histogram(df, x='Wait Time Ordering (min)', nbins=20, title='Distribution of Wait Times at Order Window')
    fig_wait_payment = px.histogram(df, x='Wait Time Payment (min)', nbins=20, title='Distribution of Wait Times at Payment Window')
    fig_total = px.histogram(df, x='Total Time (min)', nbins=20, title='Distribution of Total Time in System')

    return results, fig_wait_order,fig_wait_payment, fig_total, df

# --- Streamlit App ---
st.set_page_config(page_title="Drive-Through Simulation", page_icon=":car:", layout="wide")
st.title("Drive-Through Simulation")
st.write("""
This app simulates a drive-through service using SimPy.
Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.
""")

# --- Sidebar (Inputs) ---
with st.sidebar:
    st.header("Simulation Parameters")

    # Initialize session state variables if they don't exist
    if 'arrival_rate' not in st.session_state:
        st.session_state.arrival_rate = 1.0
    if 'order_time' not in st.session_state:
        st.session_state.order_time = 1.0
    if 'prep_time' not in st.session_state:
        st.session_state.prep_time = 400.0 / 60.0
    if 'payment_time' not in st.session_state:
        st.session_state.payment_time = 1.0
    if 'order_queue_capacity' not in st.session_state:
        st.session_state.order_queue_capacity = 5
    if 'service_queue_capacity' not in st.session_state:
        st.session_state.service_queue_capacity = 8
    if 'simulation_time' not in st.session_state:
        st.session_state.simulation_time = 600
    if 'num_order_stations' not in st.session_state:
        st.session_state.num_order_stations = 2

    # Use st.session_state to store and retrieve widget values
    arrival_rate = st.number_input("Arrival Rate (cars/min)", min_value=0.1, max_value=10.0, value=st.session_state.arrival_rate, step=0.1, format="%.1f", key="arrival_rate")
    order_time = st.number_input("Order Time (min)", min_value=0.1, max_value=10.0, value=st.session_state.order_time, step=0.1, format="%.1f", key="order_time")
    prep_time = st.number_input("Preparation Time (min)", min_value=0.1, max_value=20.0, value=st.session_state.prep_time, step=0.1, format="%.2f", key="prep_time")
    payment_time = st.number_input("Payment Time (min)", min_value=0.1, max_value=5.0, value=st.session_state.payment_time, step=0.1, format="%.1f", key="payment_time")
    order_queue_capacity = st.number_input("Order Queue Capacity", min_value=1, max_value=100, value=st.session_state.order_queue_capacity, step=1, key="order_queue_capacity")
    service_queue_capacity = st.number_input("Service Queue Capacity", min_value=1, max_value=100, value=st.session_state.service_queue_capacity, step=1, key="service_queue_capacity")
    simulation_time = st.number_input("Simulation Time (min)", min_value=1, max_value=1440, value=st.session_state.simulation_time, step=1, key="simulation_time")
    num_order_stations = st.number_input("Number of Order Stations", min_value=1, max_value=10, value=st.session_state.num_order_stations, step=1, key="num_order_stations")

    if st.button("Run Simulation"):
        config = Config(arrival_rate, order_time, prep_time, payment_time, order_queue_capacity, service_queue_capacity, simulation_time, num_order_stations)
        metrics = run_simulation(config)
        results, fig_wait_order, fig_wait_payment, fig_total, df = analyze_results(metrics, config)

# --- Main Area (Results) ---
st.header("Simulation Results")

if 'metrics' in locals():
    st.dataframe(df)

    # Display metrics in columns for better layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cars Served", results['Cars Served'])
        st.metric("Cars Blocked (Order Queue)", results['Cars Blocked (Order Queue)'])
        st.metric("Cars Blocked (Service Queue)", results['Cars Blocked (Service Queue)'])
    with col2:
        st.metric("Throughput (cars/hour)", results['Throughput (cars/hour)'])
        st.metric("Avg Wait Ordering (min)", results['Avg Wait Ordering (min)'])
        st.metric("Avg Wait Payment (min)", results['Avg Wait Payment (min)'])
    with col3:
        st.metric("Avg Wait Before Order Queue (min)", results
