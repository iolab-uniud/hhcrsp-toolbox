import plotly.express as px
import plotly.colors as colors
import numpy as np
import pandas as pd
from ..models import Instance, Solution
import re
import datetime as dt

def visualize(instance: Instance, solution: Solution):        
    def sub_label(label):
        m = re.match('(\w)(\d+)', label)
        if not m:
            return label
        return f"{m.group(1)}<sub>{m.group(2)}</sub>"

    scale = colors.sample_colorscale('viridis', len(instance.caregivers))
    np.random.shuffle(scale)
    c_dict = {f"{sub_label(c.id)}": scale[i] for i, c in enumerate(instance.caregivers)}
    c_dict['time windows'] = 'rgb(0.2, 0.2, 0.2)'

    ref_date = dt.datetime(2020, 1, 1)

    time_windows = []
    total_services = sum(len(p.required_services) for p in instance.patients)
    for p in instance.patients:
        if len(p.required_services) == 1:
            time_windows.append(dict(
                start=ref_date + dt.timedelta(minutes=p.time_window[0]), 
                end=ref_date + dt.timedelta(minutes=p.time_window[1]), 
                y=sub_label(p.id), 
                caregiver='time windows', 
                hover=f'Time window for {p.id} {p.time_window}', 
                text="", hatch='on-time', legend=False
            ))
        else:
            for i in range(len(p.required_services)):
                time_windows.append(dict(
                    start=ref_date + dt.timedelta(minutes=p.time_window[0]), 
                    end=ref_date + dt.timedelta(minutes=p.time_window[1]), 
                    y=f'{sub_label(p.id)}<sup>{i + 1}</sup>', 
                    caregiver='time windows', hover=f'Time window for {p.id} {p.time_window}', 
                    text="", hatch='on-time', legend=False
                ))

    waiting_times = []
    services = []
    annotations = []
    for r in solution.routes:
        prev = None
        for visit in r._visits:
            p = instance._patients[visit.patient]            
            if len(p.required_services) == 1:
                services.append(dict(
                    start=ref_date + dt.timedelta(minutes=visit.start_service_time), 
                    end=ref_date + dt.timedelta(minutes=visit.end_service_time), 
                    y=sub_label(visit.patient), 
                    caregiver=sub_label(r.caregiver_id), 
                    hover=f'Visit for service {sub_label(visit.service)} by caregiver {sub_label(r.caregiver_id)} {(visit.start_service_time, visit.end_service_time)}', 
                    text=f"{sub_label(visit.service)} / {sub_label(r.caregiver_id)}", 
                    hatch='on-time' if visit.start_service_time <= p.time_window[1] else 'late'
                ))          
                cur = (ref_date + dt.timedelta(minutes=visit.arrival_at_patient), sub_label(visit.patient))     
                if visit.start_service_time > visit.arrival_at_patient:
                    waiting_times.append(dict(
                        start=ref_date + dt.timedelta(minutes=visit.arrival_at_patient), 
                        end=ref_date + dt.timedelta(minutes=visit.start_service_time), 
                        y=sub_label(visit.patient), caregiver=sub_label(r.caregiver_id), 
                        hover=f'Waiting time for {sub_label(visit.patient)}', 
                        text="", 
                        hatch='waiting'
                    ))
            else:
                i = next(i for i, s in enumerate(p.required_services) if s.service == visit.service)
                services.append(dict(
                    start=ref_date + dt.timedelta(minutes=visit.start_service_time), 
                    end=ref_date + dt.timedelta(minutes=visit.end_service_time), 
                    y=f'{sub_label(p.id)}<sup>{i + 1}</sup>', 
                    caregiver=sub_label(r.caregiver_id), 
                    hover=f'Visit for service {sub_label(visit.service)} by caregiver {sub_label(r.caregiver_id)} {(visit.start_service_time, visit.end_service_time)}',
                    text=f"{sub_label(visit.service)} / {sub_label(r.caregiver_id)}",
                    hatch='on-time' if visit.start_service_time <= p.time_window[1] else 'late'
                ))
                cur = (ref_date + dt.timedelta(minutes=visit.arrival_at_patient), f'{sub_label(p.id)}<sup>{i + 1}</sup>')
                if visit.start_service_time > visit.arrival_at_patient:
                    waiting_times.append(dict(
                        start=ref_date + dt.timedelta(minutes=visit.arrival_at_patient), 
                        end=ref_date + dt.timedelta(minutes=visit.start_service_time), 
                        y=f'{sub_label(p.id)}<sup>{i + 1}</sup>', 
                        caregiver=sub_label(r.caregiver_id), 
                        hover=f'Waiting time for {sub_label(visit.patient)}', 
                        text="", 
                        hatch='waiting'
                    ))
            if prev:
                annotations.append(dict(ax=prev[0], ay=prev[1], axref="x", ayref="y", x=cur[0], y=cur[1], xref="x", yref="y", arrowsize=1, arrowwidth=2, arrowside='end', arrowhead=4, arrowcolor=c_dict[sub_label(r.caregiver_id)], opacity=0.5))
            prev = (ref_date + dt.timedelta(minutes=visit.end_service_time), cur[1])

    df = pd.DataFrame(
        time_windows + 
        services +
        waiting_times
    )
    
    fig = px.timeline(df, x_start='start', x_end='end', y='y', 
                      color='caregiver', color_discrete_map=c_dict, opacity=0.5, 
                      hover_name='hover', hover_data={'start': False, 'end': False, 'y': False, 'caregiver': False, 'hover': False, 'text': False, 'hatch': False},
                      pattern_shape='hatch', pattern_shape_map={'waiting': 'x', 'on-time': '', 'late': '\\'},
                      labels={'y': 'Patient', 'x': 'Time'}, text='text')     

    fig.update_xaxes(tickformat="%H:%M", title_text="Time", range=[ref_date, ref_date + dt.timedelta(minutes=max(r.locations[-1].arrival_time for r in solution.routes))])
    fig.update_yaxes(showgrid=True, range=[-1, total_services + 1])

    fig = fig.update_traces(marker=dict(line=dict(width=1, color='black')))   
    fig.update_layout(height=25 * (total_services), 
                      showlegend=True,
                      legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, title_text="Caregiver"), 
                      barmode='stack')
    
    
    for d in fig.data:
        if d.hovertext[0].startswith('Time window'):
            d.showlegend = False

    fig.update_layout(updatemenus=[
        dict(type="buttons",
             active=1,
             buttons=[
                dict(label="Show Routes",
                     method="relayout",
                     args=["annotations", annotations]),
                dict(label="Hide Routes",
                     method="relayout",
                     args=["annotations", []])
          ])])
    return fig