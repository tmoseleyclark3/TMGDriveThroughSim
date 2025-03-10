import streamlit as st
import simpy
import random
import pandas as pd
import plotly.express as px
import numpy as np  # Import numpy

# --- Simulation Code (from previous working versions) ---
class Config:
    def __init__(self, arrival_rate, order_time, prep_time, payment_time, queue_capacity, simulation_time):
        self.ARRIVAL_RATE = arrival_rate
        self.ORDER_TIME = order_time
        self.PREP_TIME = prep_time
        self.PAYMENT_TIME = payment_time
        self.QUEUE_CAPACITY = queue_capacity
        self.SIMULATION_TIME = simulation_time

class DriveThrough:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.order_station = simpy.Resource(env, capacity=1)
        self.service_window = simpy.Resource(env, capacity=1)
        self.service_queue = simpy.Store(env, capacity=config.QUEUE_CAPACITY)
        self.order_prep = simpy.Resource(env, capacity=1)
        self.order_ready_events = {}
        self.metrics = {  # Add car_id here
            'wait_times_ordering': [],
            'wait_times_service': [],
            'wait_times_before_service': [],
            'total_times': [],
            'cars_served': 0,
            'cars_blocked': 0,
            'car_ids': []  # Store car IDs
        }

    def process_car(self, car_id):
        arrival_time = self.env.now
        self.metrics['car_ids'].append(car_id)  # Track car ID

        # Add placeholders for ALL metrics at the BEGINNING:
        self.metrics['wait_times_ordering'].append(np.nan)
        self.metrics['wait_times_before_service'].append(np.nan)
        self.metrics['wait_times_service'].append(np.nan)
        self.metrics['total_times'].append(np.nan)

        # Stage 1: Ordering
        with self.order_station.request() as request:
            yield request
            order_start_time = self.env.now
            yield self.env.timeout(self.config.ORDER_TIME)
            order_end_time = self.env.now
            # Overwrite the placeholder with the actual value:
            self.metrics['wait_times_ordering'][-1] = order_end_time - order_start_time

        # Stage 2: Start order prep (non-blocking)
        self.env.process(self.prep_order(car_id))

        # Stage 3: Service Queue (with blocking)
        enter_service_queue_time = self.env.now
        try:
            yield self.service_queue.put(car_id)
            # Overwrite the placeholder:
            self.metrics['wait_times_before_service'][-1] = self.env.now - enter_service_queue_time

            # Stage 4: Payment and Handoff
            with self.service_window.request() as request:
                yield request
                yield self.env.timeout(self.config.PAYMENT_TIME)
                service_end_time = self.env.now
                #Overwrite placeholder
                self.metrics['wait_times_service'][-1] = service_end_time - enter_service_queue_time
                yield self.service_queue.get()

            # Stage 5: Wait for order prep
            yield self.order_ready_events[car_id]
            del self.order_ready_events[car_id]

            # Completion
            total_time = self.env.now - arrival_time
            # Overwrite the placeholder:
            self.metrics['total_times'][-1] = total_time
            self.metrics['cars_served'] += 1

        except simpy.Interrupt:
            self.metrics['cars_blocked'] += 1
            # No need to add np.nan again; it's already there as a placeholder
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
    env = simpy.Environment()
    drive_through = DriveThrough(env, config)
    env.process(car_arrivals(env, drive_through))
    env.run(until=config.SIMULATION_TIME)
    return drive_through.metrics

def analyze_results(metrics, config):

    if not metrics['wait_times_ordering']:  # Handle empty metrics
        return {
            'Cars Served': 0,
            'Cars Blocked': 0,
            'Throughput (cars/hour)': 0.0,
            'Avg Wait Ordering (min)': 0.0,
            'Avg Wait Before Service (min)': 0.0,
            'Avg Wait Service (min)': 0.0,
            'Avg Total Time (min)': 0.0,
        }, px.histogram(), px.histogram(), pd.DataFrame() # Return empty figures and df


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

    run_button = st.button("Run Simulation")

# --- Main Content (Outputs) ---
if run_button:
    with st.spinner("Running simulation..."):
        config = Config(arrival_rate, order_time, prep_time, payment_time, queue_capacity, simulation_time)
        metrics = run_simulation(config)
        results, fig_wait, fig_total, df = analyze_results(metrics, config)

        # --- Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Cars Served", results['Cars Served'])
        col2.metric("Cars Blocked", results['Cars Blocked'])
        col3.metric("Throughput (cars/hour)", results['Throughput (cars/hour)'])

        # --- Data Table ---
        st.subheader("Wait Times (minutes)")
        st.dataframe(pd.DataFrame([results]).drop(columns=['Cars Served','Cars Blocked','Throughput (cars/hour)']), use_container_width=True)

        # --- Plots ---
        st.plotly_chart(fig_wait, use_container_width=True)
        st.plotly_chart(fig_total, use_container_width=True)

        # --- Raw Data (Optional) ---
        with st.expander("Show Raw Data"):
            st.dataframe(df, use_container_width=True)  # Show the full DataFrame

else:
    st.write("Adjust the parameters and click 'Run Simulation'.") # Message Before
