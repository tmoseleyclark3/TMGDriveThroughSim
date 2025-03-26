import streamlit as st
import simpy
# import random # No longer needed for the core simulation logic
import pandas as pd
import plotly.express as px
import numpy as np # Still used in analysis

# --- Configuration Class (Unchanged Structure, Meaning Adjusted) ---
class Config:
    def __init__(self, cars_per_hour, order_time, prep_time, payment_time, order_queue_capacity, payment_queue_capacity, simulation_time):
        self.CARS_PER_HOUR = cars_per_hour
        # Times are now EXACT durations
        self.ORDER_TIME = order_time
        self.PREP_TIME = prep_time
        self.PAYMENT_TIME = payment_time
        self.ORDER_QUEUE_CAPACITY = order_queue_capacity
        self.PAYMENT_QUEUE_CAPACITY = payment_queue_capacity
        self.SIMULATION_TIME = simulation_time
        # Calculate FIXED inter-arrival time in minutes
        if self.CARS_PER_HOUR > 0:
             self.INTERARRIVAL_TIME = 60.0 / self.CARS_PER_HOUR
        else:
             self.INTERARRIVAL_TIME = float('inf') # Avoid division by zero, effectively no arrivals


# --- Drive-Through Simulation Class (Revised for Determinism) ---
class DriveThrough:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # Resources (single server for each)
        self.order_station = simpy.Resource(env, capacity=1)
        self.payment_window = simpy.Resource(env, capacity=1)
        self.order_prep_station = simpy.Resource(env, capacity=1)

        # Queues
        self.order_queue = simpy.Store(env)
        self.payment_queue = simpy.Store(env)

        # Events for order preparation completion
        self.order_ready_events = {} # Key: car_id, Value: env.event()

        # Simplified Metrics Collection
        self.metrics = {
            'car_records': [], # List to store dicts for each car's journey
            'cars_completed': 0,
            'cars_turned_away_arrival': 0, # Renamed from balked
            'cars_blocked_order': 0,
            'cars_blocked_payment': 0,
        }
        self.car_counter = 0

    def car_process(self, car_id):
        """Process for each car - NOW DETERMINISTIC."""
        arrival_time = self.env.now
        print(f"{self.env.now:.2f}: Car {car_id} arrived.")

        # Record for this car
        record = {'car_id': car_id, 'arrival_time': arrival_time, 'status': 'arrived'}

        # --- Stage 0/1: Check Order Queue Capacity (Deterministic Blocking) ---
        order_queue_len = len(self.order_queue.items)
        if order_queue_len >= self.config.ORDER_QUEUE_CAPACITY:
            # Blocking: Queue is strictly full, no space. Car must leave.
            # Removed the random balking component.
            print(f"{self.env.now:.2f}: Car {car_id} TURNED AWAY (Order queue full: {order_queue_len}/{self.config.ORDER_QUEUE_CAPACITY}).")
            self.metrics['cars_turned_away_arrival'] += 1 # Use the new metric name
            record['status'] = 'turned_away_arrival'
            record['finish_time'] = self.env.now
            self.metrics['car_records'].append(record)
            return # Car leaves
        else:
            # Enter the order queue
            queue_entry_time = self.env.now
            record['order_queue_entry_time'] = queue_entry_time
            order_queue_put_req = self.order_queue.put(car_id) # Request to put in store
            yield order_queue_put_req
            print(f"{self.env.now:.2f}: Car {car_id} entered order queue (length: {len(self.order_queue.items)}).")

        # --- Stage 2: Ordering at Station (Fixed Time) ---
        with self.order_station.request() as order_req:
            yield order_req # Wait for order station

            yield self.order_queue.get() # Get car from order queue
            order_start_time = self.env.now
            record['order_start_time'] = order_start_time
            # Ensure entry time exists before calculating wait
            if 'order_queue_entry_time' in record:
                 record['order_queue_wait'] = order_start_time - record['order_queue_entry_time']
            else:
                 record['order_queue_wait'] = 0 # Should not happen if logic is correct

            print(f"{self.env.now:.2f}: Car {car_id} started ordering.")

            order_duration = self.config.ORDER_TIME # Use EXACT time
            yield self.env.timeout(order_duration) # Ordering process

            order_end_time = self.env.now
            record['order_end_time'] = order_end_time
            record['order_duration'] = order_end_time - order_start_time
            print(f"{self.env.now:.2f}: Car {car_id} finished ordering.")

        # Start order preparation (runs in parallel)
        self.order_ready_events[car_id] = self.env.event()
        self.env.process(self.prepare_order(car_id, record))

        # --- Stage 3: Check Payment Queue Capacity (Deterministic Blocking) ---
        payment_queue_len = len(self.payment_queue.items)
        if payment_queue_len >= self.config.PAYMENT_QUEUE_CAPACITY:
            # Blocking: Payment queue is strictly full. Car must leave.
            print(f"{self.env.now:.2f}: Car {car_id} BLOCKED (Payment queue full: {payment_queue_len}/{self.config.PAYMENT_QUEUE_CAPACITY}).")
            self.metrics['cars_blocked_payment'] += 1
            record['status'] = 'blocked_payment'
            record['finish_time'] = self.env.now
            if car_id in self.order_ready_events: del self.order_ready_events[car_id]
            self.metrics['car_records'].append(record)
            return # Car leaves
        else:
            # Enter the payment queue
            queue_entry_time = self.env.now
            record['payment_queue_entry_time'] = queue_entry_time
            payment_queue_put_req = self.payment_queue.put(car_id)
            yield payment_queue_put_req
            print(f"{self.env.now:.2f}: Car {car_id} entered payment queue (length: {len(self.payment_queue.items)}).")

        # --- Stage 4: Payment at Window (Fixed Time) ---
        with self.payment_window.request() as payment_req:
            yield payment_req # Wait for payment window

            yield self.payment_queue.get() # Get car from payment queue
            payment_start_time = self.env.now
            record['payment_start_time'] = payment_start_time
             # Ensure entry time exists before calculating wait
            if 'payment_queue_entry_time' in record:
                 record['payment_queue_wait'] = payment_start_time - record['payment_queue_entry_time']
            else:
                  record['payment_queue_wait'] = 0 # Should not happen

            print(f"{self.env.now:.2f}: Car {car_id} started payment.")

            payment_duration = self.config.PAYMENT_TIME # Use EXACT time
            yield self.env.timeout(payment_duration) # Payment process

            payment_end_time = self.env.now
            record['payment_end_time'] = payment_end_time
            record['payment_duration'] = payment_end_time - payment_start_time
            print(f"{self.env.now:.2f}: Car {car_id} finished payment.")

        # --- Stage 5: Wait for Order Prep Completion (if necessary) ---
        if car_id in self.order_ready_events:
            prep_event = self.order_ready_events[car_id]
            if not prep_event.triggered:
                 print(f"{self.env.now:.2f}: Car {car_id} waiting for order preparation...")
                 yield prep_event
            record['pickup_time'] = self.env.now
            print(f"{self.env.now:.2f}: Car {car_id} received order.")
            del self.order_ready_events[car_id]
        else:
             print(f"{self.env.now:.2f}: WARNING - Car {car_id} has no order ready event!")
             record['pickup_time'] = self.env.now

        # --- Stage 6: Service Completion ---
        finish_time = self.env.now
        record['finish_time'] = finish_time
        record['total_time'] = finish_time - arrival_time
        record['status'] = 'completed'
        self.metrics['cars_completed'] += 1
        self.metrics['car_records'].append(record)
        print(f"{self.env.now:.2f}: Car {car_id} completed service. Total time: {record['total_time']:.2f}")


    def prepare_order(self, car_id, record):
        """Simulates order preparation - NOW DETERMINISTIC."""
        with self.order_prep_station.request() as prep_req:
            yield prep_req # Request order prep station
            prep_start_time = self.env.now
            record['prep_start_time'] = prep_start_time
            print(f"{self.env.now:.2f}: Order preparation started for car {car_id}.")

            prep_duration = self.config.PREP_TIME # Use EXACT time
            yield self.env.timeout(prep_duration) # Order prep time

            prep_end_time = self.env.now
            record['prep_end_time'] = prep_end_time
            record['prep_duration'] = prep_end_time - prep_start_time
            print(f"{self.env.now:.2f}: Order preparation finished for car {car_id}.")

            if car_id in self.order_ready_events:
                 self.order_ready_events[car_id].succeed()

    def car_arrival_process(self):
        """Generates car arrivals at FIXED intervals."""
        # Don't start the first car exactly at time 0, wait one interval,
        # Or start at 0 if you prefer. Let's wait one interval.
        if self.config.INTERARRIVAL_TIME == float('inf'):
             yield self.env.timeout(self.config.SIMULATION_TIME) # Wait forever if rate is 0
             return

        yield self.env.timeout(self.config.INTERARRIVAL_TIME) # Initial delay

        while True:
            self.car_counter += 1
            self.env.process(self.car_process(self.car_counter))

            # Wait for the fixed interarrival time before scheduling the next
            yield self.env.timeout(self.config.INTERARRIVAL_TIME)


def run_simulation(config):
    """Runs the drive-through simulation."""
    print("\n--- Starting DETERMINISTIC Simulation ---") # Added Deterministic keyword
    print("Configuration:")
    for param, value in config.__dict__.items():
        # Format INTERARRIVAL_TIME for better readability if it's not infinite
        if param == 'INTERARRIVAL_TIME' and value != float('inf'):
             print(f"  {param}: {value:.4f}")
        else:
             print(f"  {param}: {value}")


    env = simpy.Environment()
    drive_through = DriveThrough(env, config)
    env.process(drive_through.car_arrival_process())
    env.run(until=config.SIMULATION_TIME)

    print(f"--- Simulation Complete at time {env.now:.2f} ---")
    return drive_through.metrics

# --- Analysis Function (Updated for Renamed Metric) ---
def analyze_results(metrics, config):
    """Analyzes simulation metrics and generates results."""
    records = metrics['car_records']
    if not records:
        st.warning("No cars entered the system during the simulation.")
        return {
             'summary': {'Total Cars Generated': 0},
             'dataframe': pd.DataFrame(),
             'plots': {}
        }

    df = pd.DataFrame(records)

    # --- Calculate Summary Statistics ---
    total_generated = metrics['car_counter'] # Use counter for total attempts
    completed_df = df[df['status'] == 'completed'].copy()

    summary = {
        'Total Cars Attempted': total_generated, # Changed label
        'Cars Completed Service': metrics['cars_completed'],
        'Cars Turned Away (Arrival)': metrics['cars_turned_away_arrival'], # Updated Key
        'Cars Blocked (Order Queue)': metrics['cars_blocked_order'], # Note: This might become 0 if turned away earlier
        'Cars Blocked (Payment Queue)': metrics['cars_blocked_payment'],
        'Throughput (cars/hour)': 0.0,
        'Avg Order Queue Wait (min)': 0.0, # Note: 'Avg' might be misleading in deterministic, but std dev will be 0
        'Avg Payment Queue Wait (min)': 0.0,
        'Avg Total Time (min)': 0.0,
        'Avg Order Duration (min)': 0.0,
        'Avg Prep Duration (min)': 0.0,
        'Avg Payment Duration (min)': 0.0,
    }

    if metrics['cars_completed'] > 0:
        summary['Throughput (cars/hour)'] = metrics['cars_completed'] / config.SIMULATION_TIME * 60
        summary['Avg Order Queue Wait (min)'] = completed_df['order_queue_wait'].mean()
        summary['Avg Payment Queue Wait (min)'] = completed_df['payment_queue_wait'].mean()
        summary['Avg Total Time (min)'] = completed_df['total_time'].mean()
        summary['Avg Order Duration (min)'] = completed_df['order_duration'].mean()
        summary['Avg Prep Duration (min)'] = completed_df['prep_duration'].mean()
        summary['Avg Payment Duration (min)'] = completed_df['payment_duration'].mean()

        for key in ['Throughput (cars/hour)', 'Avg Order Queue Wait (min)',
                    'Avg Payment Queue Wait (min)', 'Avg Total Time (min)',
                    'Avg Order Duration (min)', 'Avg Prep Duration (min)', 'Avg Payment Duration (min)']:
             if pd.notna(summary[key]) and isinstance(summary[key], (int, float)):
                  summary[key] = f"{summary[key]:.2f}"
             else:
                  summary[key] = "N/A"
    else:
         # Ensure formatting even if no cars completed
         summary['Throughput (cars/hour)'] = "0.00"


    # --- Generate Plots ---
    # Plots might be less interesting in a deterministic simulation (potentially showing single bars or points)
    # But we keep them for consistency
    plots = {}
    if not completed_df.empty:
        # Explicitly set bins or use bar charts if data is likely discrete/single-valued
        plots['fig_wait_order_queue'] = px.histogram(completed_df, x='order_queue_wait', title='Distribution of Order Queue Wait Times (Completed Cars)')
        plots['fig_wait_payment_queue'] = px.histogram(completed_df, x='payment_queue_wait', title='Distribution of Payment Queue Wait Times (Completed Cars)')
        plots['fig_total_time'] = px.histogram(completed_df, x='total_time', title='Distribution of Total System Time (Completed Cars)')
    else:
         plots['fig_wait_order_queue'] = px.histogram(title='Distribution of Order Queue Wait Times (No Completed Cars)')
         plots['fig_wait_payment_queue'] = px.histogram(title='Distribution of Payment Queue Wait Times (No Completed Cars)')
         plots['fig_total_time'] = px.histogram(title='Distribution of Total System Time (No Completed Cars)')

    # Ensure all expected columns exist
    expected_cols = ['car_id', 'arrival_time', 'status', 'finish_time',
                     'order_queue_entry_time', 'order_start_time', 'order_queue_wait',
                     'order_end_time', 'order_duration',
                     'payment_queue_entry_time', 'payment_start_time', 'payment_queue_wait',
                     'payment_end_time', 'payment_duration',
                     'prep_start_time', 'prep_end_time', 'prep_duration', 'pickup_time', 'total_time']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    df = df[expected_cols] # Reorder

    return {'summary': summary, 'dataframe': df, 'plots': plots}

# --- Streamlit App (Updated Labels) ---
st.set_page_config(page_title="Deterministic Drive-Through Simulation", page_icon=":gear:", layout="wide") # Changed icon
st.title("Deterministic Drive-Through Simulation") # Changed title
st.write("""
This app simulates a drive-through service with **fixed** parameters (no randomness).
Arrivals occur at a constant rate, and service times are exact. Adjust parameters and run the simulation.
""")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Simulation Parameters (Exact Values)") # Changed header

    defaults = {
        'cars_per_hour': 70.0, 'order_time': 1.2, 'prep_time': 2.0, 'payment_time': 0.8,
        'order_queue_capacity': 15, 'payment_queue_capacity': 2, 'simulation_time': 60
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Input widgets - Updated labels to remove "Avg"
    st.session_state.cars_per_hour = st.number_input("Cars per Hour (Rate)", min_value=0.0, max_value=300.0, value=st.session_state.cars_per_hour, step=1.0, format="%.1f", key="cars_per_hour_input")
    st.session_state.order_time = st.number_input("Order Time (min)", min_value=0.01, max_value=10.0, value=st.session_state.order_time, step=0.1, format="%.2f", key="order_time_input") # Allow slightly smaller min
    st.session_state.prep_time = st.number_input("Preparation Time (min)", min_value=0.01, max_value=20.0, value=st.session_state.prep_time, step=0.1, format="%.2f", key="prep_time_input")
    st.session_state.payment_time = st.number_input("Payment Time (min)", min_value=0.01, max_value=5.0, value=st.session_state.payment_time, step=0.1, format="%.2f", key="payment_time_input")
    st.session_state.order_queue_capacity = st.number_input("Order Queue Capacity (Cars)", min_value=0, max_value=100, value=st.session_state.order_queue_capacity, step=1, key="order_queue_capacity_input", help="Max cars waiting *between* arrival and order station.")
    st.session_state.payment_queue_capacity = st.number_input("Payment Queue Capacity (Cars)", min_value=0, max_value=100, value=st.session_state.payment_queue_capacity, step=1, key="payment_queue_capacity_input", help="Max cars waiting *between* ordering and payment window.")
    st.session_state.simulation_time = st.number_input("Simulation Duration (min)", min_value=1, max_value=10080, value=st.session_state.simulation_time, step=10, key="simulation_time_input", help="Total minutes to simulate.")

    run_button = st.button("Run Simulation")

# --- Main area for Results (Updated Labels) ---
st.header("Simulation Results")

if run_button:
    sim_config = Config(
        cars_per_hour=st.session_state.cars_per_hour,
        order_time=st.session_state.order_time,
        prep_time=st.session_state.prep_time,
        payment_time=st.session_state.payment_time,
        order_queue_capacity=st.session_state.order_queue_capacity,
        payment_queue_capacity=st.session_state.payment_queue_capacity,
        simulation_time=st.session_state.simulation_time
    )

    with st.spinner("Running Deterministic Simulation..."):
        raw_metrics = run_simulation(sim_config)
        analysis_results = analyze_results(raw_metrics, sim_config)

    summary = analysis_results['summary']
    df_results = analysis_results['dataframe']
    plots = analysis_results['plots']

    st.subheader("Performance Metrics") # Changed subheader
    if summary:
        cols = st.columns(4)
        cols[0].metric("Cars Completed", summary.get('Cars Completed Service', 0))
        cols[1].metric("Throughput (cars/hr)", summary.get('Throughput (cars/hour)', "N/A"))
        # Display exact times if they are constant, avg otherwise (though std dev should be 0)
        cols[2].metric("Avg Total Time (min)", summary.get('Avg Total Time (min)', "N/A"))

        cols = st.columns(4)
        cols[0].metric("Turned Away (Arrival)", summary.get('Cars Turned Away (Arrival)', 0)) # Updated label
        cols[1].metric("Blocked (Order Q)", summary.get('Cars Blocked (Order Queue)', 0))
        cols[2].metric("Blocked (Payment Q)", summary.get('Cars Blocked (Payment Queue)', 0))
        cols[3].metric("Total Attempted", summary.get('Total Cars Attempted', 0)) # Updated label

        cols = st.columns(3)
        cols[0].metric("Avg Order Q Wait (min)", summary.get('Avg Order Queue Wait (min)', "N/A"))
        cols[1].metric("Avg Payment Q Wait (min)", summary.get('Avg Payment Queue Wait (min)', "N/A"))


    st.subheader("Distributions (Completed Cars)")
    if plots:
        st.plotly_chart(plots.get('fig_total_time'), use_container_width=True)
        st.plotly_chart(plots.get('fig_wait_order_queue'), use_container_width=True)
        st.plotly_chart(plots.get('fig_wait_payment_queue'), use_container_width=True)


    st.subheader("Detailed Car Data")
    st.dataframe(df_results)


else:
    st.info("Set **exact** simulation parameters in the sidebar and click 'Run Simulation' to see results.") # Updated info text
