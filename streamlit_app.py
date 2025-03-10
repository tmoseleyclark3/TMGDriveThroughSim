import streamlit as st
import simpy
import random
import pandas as pd
import plotly.express as px
import numpy as np

# --- Simulation Code ---
class Config:
    def __init__(self, arrival_rate, order_time, prep_time, payment_time, queue_capacity, simulation_time, num_order_stations):
        self.ARRIVAL_RATE = arrival_rate
        self.ORDER_TIME = order_time
        self.PREP_TIME = prep_time
        self.PAYMENT_TIME = payment_time
        self.QUEUE_CAPACITY = queue_capacity
        self.SIMULATION_TIME = simulation_time
        self.NUM_ORDER_STATIONS = num_order_stations

class DriveThrough:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.order_stations = [simpy.Resource(env, capacity=1) for _ in range(config.NUM_ORDER_STATIONS)]
        self.service_window = simpy.Resource(env, capacity=1)
        self.service_queue = simpy.Store(env, capacity=config.QUEUE_CAPACITY)
        self.order_prep = simpy.Resource(env, capacity=1)
        self.order_ready_events = {}
        self.metrics = {
            'wait_times_ordering': [],
            'wait_times_service': [],
            'wait_times_before_service': [],
            'total_times': [],
            'cars_served': 0,
            'cars_blocked': 0,
            'car_ids': []
        }

    def process_car(self, car_id):
        print(f"Car {car_id} arrived at {self.env.now}")
        arrival_time = self.env.now
        self.metrics['car_ids'].append(car_id)

        self.metrics['wait_times_ordering'].append(np.nan)
        self.metrics['wait_times_before_service'].append(np.nan)
        self.metrics['wait_times_service'].append(np.nan)
        self.metrics['total_times'].append(np.nan)

        # Stage 1: Ordering (multiple stations)
        order_station = random.choice(self.order_stations)
        with order_station.request() as request:
            yield request
            order_start_time = self.env.now
            yield self.env.timeout(self.config.ORDER_TIME)
            order_end_time = self.env.now
            self.metrics['wait_times_ordering'][-1] = order_end_time - order_start_time
            print(f"Car {car_id} finished ordering at {self.env.now}")

        # Stage 2: Start order prep (non-blocking)
        self.env.process(self.prep_order(car_id))

        # Stage 3: Service Queue (with blocking)
        enter_service_queue_time = self.env.now
        try:
            yield self.service_queue.put(car_id)
            self.metrics['wait_times_before_service'][-1] = self.env.now - enter_service_queue_time
            print(f"Car {car_id} entered service queue at {self.env.now}")

            # Stage 4: Payment and Handoff
            with self.service_window.request() as request:
                yield request
                service_start_time = self.env.now
                yield self.env.timeout(self.config.PAYMENT_TIME)
                service_end_time = self.env.now
                self.metrics['wait_times_service'][-1] = service_end_time - service_start_time
                yield self.service_queue.get()
                print(f"Car {car_id} finished service at {self.env.now}")

            # Stage 5: Wait for order prep
            yield self.order_ready_events[car_id]
            del self.order_ready_events[car_id]
            print(f"Car {car_id} order ready at {self.env.now}")

            # Completion
            total_time = self.env.now - arrival_time
            self.metrics['total_times'][-1] = total_time
            self.metrics['cars_served'] += 1
            print(f"Car {car_id} completed at {self.env.now}")

        except simpy.Interrupt:
            self.metrics['cars_blocked'] += 1
            print(f"Car {car_id} blocked at {self.env.now}")
            return

    def prep_order(self, car_id):
        with self.order_prep.request() as req:
            yield req
            yield self.env.timeout(self.config.PREP_TIME)
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
    if not metrics['wait_times_ordering']:
        return {
            'Cars Served': 0,
            'Cars Blocked': 0,
            'Throughput (cars/hour)': 0.0,
            'Avg Wait Ordering (min)': 0.0,
            'Avg Wait Before Service (min)': 0.0,
            'Avg Wait Service (min)': 0.0,
            'Avg Total Time (min)': 0.0,
        }, px.histogram(), px.histogram(), pd.DataFrame()

    df = pd.DataFrame({
        'Car ID': metrics['car_ids'],
        'Wait Time Ordering (min)': metrics['wait_times_ordering'],
        'Wait Time Before Service (min)': metrics['wait_times_before_service'],
        'Wait Time Service (min)': metrics['wait_times_service'],
        'Total Time (min)': metrics['total_times']
    })

    avg_wait_ordering = df['Wait Time Ordering (min)'].mean()
    avg_wait_before_service = df['Wait Time Before Service (min)'].mean()
    avg_wait_service = df['Wait Time Service (min)'].mean()
    avg_total_time = df['Total Time (min)'].mean()
    throughput = metrics['cars_served'] / config.SIMULATION_TIME * 60

    results = {
        'Cars Served': metrics['cars_served'],
        'Cars Blocked': metrics['cars_blocked'],
        'Throughput (cars/hour)': f"{throughput:.2f}",
        'Avg Wait Ordering (min)': f"{avg_wait_ordering:.2f}",
        'Avg Wait Before Service (min)': f"{avg_wait_before_service:.2f}",
        'Avg Wait Service (min)': f"{avg_wait_service:.2f}",
        'Avg Total Time (min)': f"{avg_total_time:.2f}",
    }

    fig_wait = px.histogram(df, x='Wait Time Service (min)', nbins=20, title='Distribution of Wait Times at Service Window')
    fig_total = px.histogram(df, x='Total Time (min)', nbins=20, title='Distribution of Total Time in System')

    return results, fig_wait, fig_total, df

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
    arrival_rate = st.number_input("Arrival Rate (cars/min)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    order_time = st.number_input("Order Time (min)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    prep_time = st.number_input("Preparation Time (min)", min_value=0.1, max_value=20.0, value=400.0 / 60.0, step=0.1, format="%.2f")
    payment_time = st.number_input("Payment Time (min)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f")
    queue_capacity = st.number_input("Queue Capacity", min_value=1, max_value=100, value=8, step=1)
    simulation_time = st.number_input("Simulation Time (min)", min_value=1, max_value=1440, value=600, step=1)
    num_order_stations = st.number_input("Number of Order Stations", min_value=1, max_value=10, value=2, step=1) # added num_order_stations

    run_button = st.button("Run Simulation")
    
# --- DriveThrough Class ---
def prep_order(self, car_id):
    with self.order_prep.request() as req:
        yield req
        yield self.env.timeout(self.config.PREP_TIME) # PREP_TIME is in minutes
        self.order_ready_events[car_id].succeed()
