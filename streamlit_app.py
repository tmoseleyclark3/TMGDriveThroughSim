import streamlit as st
import simpy
import random
import pandas as pd
import plotly.express as px
import numpy as np

# --- Simulation Code --- (No changes here)
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
        self.order_prep = simpy.Resource(env, capacity=1)  # Could be multiple prep stations
        self.order_ready_events = {}
        self.metrics = {
            'wait_times_ordering': [],
            'wait_times_service': [],
            'wait_times_before_service': [],
            'total_times': [],
            'cars_served': 0,
            'cars_blocked': 0,
            'car_ids': [],
            'order_sizes': [],  # NEW: Track order sizes
            'balking_events': [], # NEW: Track balking
        }
        # NEW: Define a menu (item: [prep_time, probability])
        self.menu = {
            "Burger": [2.5, 0.3],  # 2.5 minutes prep time, 30% probability
            "Fries": [1.5, 0.4],
            "Salad": [2.0, 0.15],
            "Drink": [0.5, 0.8],
            "Ice Cream": [1.0, 0.35],
        }
        self.payment_types = {
            "Card": [0.7, 0.85], # [mean_time, prob_of_occurence]
            "Cash": [1.1, 0.13],
            "MobilePay": [0.5, 0.02]
        }

    def process_car(self, car_id):
        print(f"Car {car_id} arrived at {self.env.now}")
        arrival_time = self.env.now
        self.metrics['car_ids'].append(car_id)

        # Initialize metrics with NaN
        self.metrics['wait_times_ordering'].append(np.nan)
        self.metrics['wait_times_before_service'].append(np.nan)
        self.metrics['wait_times_service'].append(np.nan)
        self.metrics['total_times'].append(np.nan)
        self.metrics['order_sizes'].append(np.nan)  # Initialize order size
        self.metrics['balking_events'].append(0) # Initialize to not balked

        # --- Stage 0: Balking (NEW) ---
        # Determine if the customer balks based on queue length.
        if len(self.service_queue.items) >= self.config.QUEUE_CAPACITY * 0.8:  # Example: Balk if queue is 80% full
            balk_prob = 0.2 + (len(self.service_queue.items) / self.config.QUEUE_CAPACITY) * 0.5 # Example: increasing prob
            if random.random() < balk_prob:
                print(f"Car {car_id} balked at {self.env.now}")
                self.metrics['cars_blocked'] += 1
                self.metrics['balking_events'][-1] = 1 # Set balking event to 1
                return  # Exit the process if the car balks


        # --- Stage 1: Ordering ---
        order_station = random.choice(self.order_stations)
        with order_station.request() as request:
            yield request
            order_start_time = self.env.now

            # --- IMPROVEMENTS ---
            # 1.1. Order Size & Menu:  Determine order size and items.
            order_size = random.randint(1, 4)  # Example: 1 to 4 items per order
            self.metrics['order_sizes'][-1] = order_size  # Store the order size
            order = []
            for _ in range(order_size):
                # Choose an item based on menu probabilities.
                item = random.choices(list(self.menu.keys()), weights=[item[1] for item in self.menu.values()])[0]
                order.append(item)

            # 1.2. Variable Order Time: Order time depends on order size and complexity.
            order_time = self.config.ORDER_TIME * (0.8 + 0.4 * order_size)  # Base time + per-item time
            order_time *= random.uniform(0.9, 1.1)  # Add some random variation (+/- 10%)

            yield self.env.timeout(order_time)
            order_end_time = self.env.now
            self.metrics['wait_times_ordering'][-1] = order_end_time - order_start_time
            print(f"Car {car_id} finished ordering at {self.env.now}, Order: {order}")

        # --- Stage 2: Start order prep (non-blocking) ---
        # Pass the 'order' to the prep_order function.
        self.env.process(self.prep_order(car_id, order))

        # --- Stage 3: Service Queue ---
        enter_service_queue_time = self.env.now
        try:
            yield self.service_queue.put(car_id)
            self.metrics['wait_times_before_service'][-1] = self.env.now - enter_service_queue_time
            print(f"Car {car_id} entered service queue at {self.env.now}")

            # --- Stage 4: Payment and Handoff ---
            with self.service_window.request() as request:
                yield request
                service_start_time = self.env.now

                # --- IMPROVEMENTS ---
                # 4.1. Variable Payment Time:  Based on payment type.
                payment_type = random.choices(list(self.payment_types.keys()), weights=[pt[1] for pt in self.payment_types.values()])[0]
                payment_time = np.random.normal(loc=self.payment_types[payment_type][0], scale = self.payment_types[payment_type][0]/4 ) # Example distribution
                if payment_time < 0:
                  payment_time = self.payment_types[payment_type][0]/4
                # 4.2 Payment failure
                if payment_type == "Card":
                    if random.random() < 0.05:  # 5% chance of card failure
                        payment_time += 1.5  # extra time

                yield self.env.timeout(payment_time)


                service_end_time = self.env.now
                self.metrics['wait_times_service'][-1] = service_end_time - service_start_time
                yield self.service_queue.get()
                print(f"Car {car_id} finished service at {self.env.now}")

            # --- Stage 5: Wait for order prep ---
            yield self.order_ready_events[car_id]
            del self.order_ready_events[car_id]
            print(f"Car {car_id} order ready at {self.env.now}")

            # --- Completion ---
            total_time = self.env.now - arrival_time
            self.metrics['total_times'][-1] = total_time
            self.metrics['cars_served'] += 1
            print(f"Car {car_id} completed at {self.env.now}")

        except simpy.Interrupt:  # Should not happen with Store, but good practice.
            self.metrics['cars_blocked'] += 1
            print(f"Car {car_id} blocked at {self.env.now}") #should not happen
            return

    # Modified prep_order to accept the order list.
    def prep_order(self, car_id, order):
        with self.order_prep.request() as req:
            yield req
            total_prep_time = 0
            for item in order:
                total_prep_time += self.menu[item][0]  # Add prep time for each item
            total_prep_time *= random.uniform(0.8, 1.2) # Add random variation
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

    # Initialize session state variables if they don't exist
    if 'arrival_rate' not in st.session_state:
        st.session_state.arrival_rate = 1.0
    if 'order_time' not in st.session_state:
        st.session_state.order_time = 1.0
    if 'prep_time' not in st.session_state:
        st.session_state.prep_time = 400.0 / 60.0
    if 'payment_time' not in st.session_state:
        st.session_state.payment_time = 1.0
    if 'queue_capacity' not in st.session_state:
        st.session_state.queue_capacity = 8
    if 'simulation_time' not in st.session_state:
        st.session_state.simulation_time = 600
    if 'num_order_stations' not in st.session_state:
        st.session_state.num_order_stations = 2

    # Use st.session_state to store and retrieve widget values
    arrival_rate = st.number_input("Arrival Rate (cars/min)", min_value=0.1, max_value=10.0, value=st.session_state.arrival_rate, step=0.1, format="%.1f", key="arrival_rate")
    order_time = st.number_input("Order Time (min)", min_value=0.1, max_value=10.0, value=st.session_state.order_time, step=0.1, format="%.1f", key="order_time")
    prep_time = st.number_input("Preparation Time (min)", min_value=0.1, max_value=20.0, value=st.session_state.prep_time, step=0.1, format="%.2f", key="prep_time")
    payment_time = st.number_input("Payment Time (min)", min_value=0.1, max_value=5.0, value=st.session_state.payment_time, step=0.1, format="%.1f", key="payment_time")
    queue_capacity = st.number_input("Queue Capacity", min_value=1, max_value=100, value=st.session_state.queue_capacity, step=1, key="queue_capacity")
    simulation_time = st.number_input("Simulation Time (min)", min_value=1, max_value=1440, value=st.session_state.simulation_time, step=1, key="simulation_time")
    num_order_stations = st.number_input("Number of Order Stations", min_value=1, max_value=10, value=st.session_state.num_order_stations, step=1, key="num_order_stations")

    run_button = st.button("Run Simulation")

# --- Main Panel (Outputs) ---
if run_button:
    # Create the Config object using values from st.session_state
    config = Config(st.session_state.arrival_rate, st.session_state.order_time, st.session_state.prep_time,
                    st.session_state.payment_time, st.session_state.queue_capacity, st.session_state.simulation_time,
                    st.session_state.num_order_stations)
    metrics = run_simulation(config)
    results, fig_wait, fig_total, df = analyze_results(metrics, config)

    st.subheader("Simulation Results")
    st.write(results)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_wait, use_container_width=True)
    with col2:
        st.plotly_chart(fig_total, use_container_width=True)

    with st.expander("Raw Data"):
        st.dataframe(df)

else:
    st.write("Click 'Run Simulation' to start.")
