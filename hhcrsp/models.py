from typing import Optional, Literal, Annotated, Self, Any
from pydantic import BaseModel, Field, model_validator, computed_field, AliasChoices
from collections.abc import Sequence
import hashlib
import click
import numpy as np
import math

class DepartingPoint(BaseModel):
    id: Annotated[str, Field(min_length=1, frozen=True)]
    distance_matrix_index: Annotated[int, Field(ge=0)]
    location: tuple[float, float] = None

class Caregiver(BaseModel):
    id: Annotated[str, Field(min_length=1, frozen=True)]
    abilities: Annotated[list[str], Field(min_length=1)]
    distance_matrix_index: Annotated[int, Field(ge=0)]
    departing_point: Annotated[str, Field(min_length=1, alias=AliasChoices('starting_point_id', 'departing_point'))]
    # FIXME: working shift should be already mapped to the suitable type list(map(int, self.working_shift))
    working_shift: tuple[int, int] = None

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        if self.working_shift is not None:
            assert self.working_shift[0] < self.working_shift[1], f"Working shift not correct {self.working_shift}"
        return self

class Synchronization(BaseModel):
    type: Literal['simultaneous', 'sequential']
    distance: Optional[tuple[int, int]] = None

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        if self.type == 'sequential':
            assert self.distance[0] <= self.distance[1], f"Sequential sychronization distance not correct {self.distance}"
        return self


class Service(BaseModel):
    id: Annotated[str, Field(min_length=1, frozen=False)]
    type: Annotated[str, Field(min_length=1, frozen=False)] 
    default_duration: Optional[int] = None

class RequiredService(BaseModel):
    service: Annotated[str, Field(min_length=1)]
    duration: Optional[Annotated[int, Field(gt=0)]] = None
    actual_duration: Annotated[int, Field(gt=0, exclude=True, repr=False)] = None

class Patient(BaseModel):
    id: Annotated[str, Field(min_length=1, frozen=False)]
    required_services: Annotated[list[RequiredService], 
                                 Field(min_length=1, max_length=2, alias=AliasChoices('required_caregivers', 'required_services'))]
    distance_matrix_index: Annotated[int, Field(ge=0)]
    time_window: Optional[tuple[int, int]]
    location: Optional[tuple[float, float]] = None
    synchronization: Optional[Synchronization] = None
    incompatible_caregivers: set[str] = None

    @model_validator(mode='after')
    def _validity_checks(self) -> Self:
        if len(self.required_services) > 1:
            assert self.synchronization is not None, "Synchronization specification is mandatory if more than one caregiver is required"
        if self.time_window is not None:
            assert self.time_window[0] < self.time_window[1], f"Time window not correct {self.time_window}"
        return self
    
class Instance(BaseModel):
    name: Annotated[str, Field(min_length=1, frozen=False)]
    # FIXME: round area bounds outside the function tuple(np.around(area, decimals=4))
    area: Optional[Sequence[float]] = None
    # FIXME: compute times outside the function [[int(d.total_seconds() // 60) for d in r] for r in distances]
    distances: Sequence[Sequence[int]]
    departing_points : Sequence[DepartingPoint]
    caregivers : Sequence[Caregiver]
    patients : Sequence[Patient]
    services : Sequence[Service]

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        # Matrix Size
        expected_matrix_size = len(self.departing_points) + len(self.patients)
        assert len(self.distances) == expected_matrix_size, f"The distance matrix is supposed to have {expected_matrix_size} rows ({len(self.departing_points)} departing points + {len(self.patients)} patients)"
        for i, r in enumerate(self.distances):
            assert(len(r)) == expected_matrix_size, f"Row {i} of the distance matrix is supposed to have {expected_matrix_size} columns ({len(self.departing_points)} departing points + {len(self.patients)} patients)"
        # Foreign keys (caregivers to services and departing points)
        services_id = set(s.id for s in self.services)
        services = { s.id: s for s in self.services }
        caregivers_service_coverage = set()
        matrix_indexes = set()
        departing_points = set(dp.id for dp in self.departing_points)
        for c in self.caregivers:
            provided_services = set(c.abilities)
            assert provided_services <= set(services_id), f"Abilities of caregiver {c.id} ({provided_services - set(services_id)}) are not included in services"
            caregivers_service_coverage |= provided_services
            assert c.departing_point in departing_points, f"Departing point {c.departing_point} of caregiver {c.id} is not present in the list of departing points"
        for d in self.departing_points:
            assert d.distance_matrix_index not in matrix_indexes, f"Matrix index of departing point {d.id} is already present as a matrix index"
            matrix_indexes.add(d.distance_matrix_index)
        # Foreign keys (patients to required services), also setting 
        patients_service_requirement = set()
        for p in self.patients:
            required_services = set(s.service for s in p.required_services)
            assert required_services <= set(services_id), f"Services required by patient {p.id} ({required_services - set(services_id)}) are not included in services"
            patients_service_requirement |= required_services
            assert p.distance_matrix_index not in matrix_indexes, f"Matrix index of patient {p.id} is already present as a matrix index"
            matrix_indexes.add(p.distance_matrix_index)        
            # setting default durations if not given
            for s in p.required_services:
                s.actual_duration = s.duration or services[s.service].default_duration                
            service_types = set(services[rs].type for rs in required_services)
            assert len(service_types) == len(required_services), f"Patient {p.id} is requiring services {required_services} of the same types {service_types}"
            # Checking incompatible caregivers and getting rid of those which are not providing the required services
            if p.incompatible_caregivers:
                possible_caregivers = set(c.id for c in self.caregivers if set(c.abilities) & required_services)
                # get rid of non meaningful caregivers
                if p.incompatible_caregivers & possible_caregivers != p.incompatible_caregivers:
                    click.secho(f'Patient {p.id} has a set of incompatible caregivers which includes also caregivers not directly involved with the required services, normalizing it', fg='yellow', err=True)
                    p.incompatible_caregivers = p.incompatible_caregivers & possible_caregivers
                for s in p.required_services:
                    possible_caregivers_for_service = set(c.id for c in self.caregivers if s.service in c.abilities and c.id not in p.incompatible_caregivers)
                    assert possible_caregivers_for_service, f"Patient {p.id} has no compatible caregiver for service {s.id} (possibly because of incompatibilities or the no caregiver exists for the service)"
                
        assert patients_service_requirement <= caregivers_service_coverage, f"Some services required by patients are not provided by any caregiver ({patients_service_requirement - caregivers_service_coverage})"        
        assert matrix_indexes == set(range(expected_matrix_size)), f"Some patients / departing point have been wrongly assigned their matrix index"
        # checking thaat the time window is compatible with the service time distance in case of sequential services
        for p in self.patients:
            for s in p.required_services:
                if p.synchronization and p.synchronization.type == 'sequential':
                    assert p.time_window[1] - p.time_window[0] >= min(p.synchronization.distance), f"Patient {p.id} has a time window too short for the synchronization services required"
       
        return self
    
    @computed_field
    @property
    def signature(self) -> str:
        res = b''
        for f in self.model_fields:
            res += bytes(f"{getattr(self, f)}", 'utf-8')
        return hashlib.sha256(res).hexdigest()
    
    @property
    def features(self) -> dict[str, Any]:
        features = {}
        features['patients'] = { 
            'total': len(self.patients), 
            'single': sum(len(p.required_services) == 1 for p in self.patients) / len(self.patients), 
            'double': sum(len(p.required_services) == 2 for p in self.patients) / len(self.patients), 
            'simultaneous': sum(p.synchronization is not None and p.synchronization.type == 'simultaneous' for p in self.patients), 
            'sequential': sum(p.synchronization is not None and p.synchronization.type == 'sequential' for p in self.patients) 
        }
        features['caregivers'] = len(self.caregivers)
        features['services'] = len(self.services)
        compatible_caregivers = []
        time_windows_size = []
        service_length = []
        for p in self.patients:
            for s in p.required_services:
                possible_caregivers_for_service = set(c.id for c in self.caregivers if s.service in c.abilities and c.id not in (p.incompatible_caregivers or set()))
                compatible_caregivers.append(len(possible_caregivers_for_service))
            time_windows_size.append(p.time_window[1] - p.time_window[0])
            service_length += [s.actual_duration for s in p.required_services]
        features['compatible_caregivers'] = { 
            'min': min(compatible_caregivers), 
            'avg': sum(compatible_caregivers) / len(compatible_caregivers),
            'max': max(compatible_caregivers) 
        }
        features['time_windows_size'] = { 'min': min(time_windows_size), 'avg': sum(time_windows_size) / len(self.patients), 'max': max(time_windows_size) }
        features['service_length'] = { 'min': min(service_length), 'avg': sum(service_length) / len(service_length), 'max': max(service_length) }
        distances = np.array(self.distances)
        mask = np.ones(distances.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        features['distances'] = { 
            'min': float(distances[mask].min()), 
            'avg': float(distances[mask].mean()), 
            'max': float(distances[mask].max()) 
        }
        return features
    

class RouteLocation(BaseModel):
    arrival_time: Annotated[int | float, Field(ge=0)]
    departure_time: Annotated[int | float, Field(ge=0)]
    patient: str
    service: str

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        assert self.arrival_time < self.departure_time, "Arrival time is greater than departure time"
        return self

class CaregiverRoute(BaseModel):
    caregiver_id: str
    locations: list[RouteLocation]

class Solution(BaseModel):
    instance: Optional[Annotated[str, Field(min_length=1)]] = None
    routes: list[CaregiverRoute]

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        assert len(self.routes) == len(set(r.caregiver_id for r in self.routes)), "Some caregivers are repeated in the solution"
        for r in self.routes:
            for i, l in enumerate(r.locations):
                if i > 0:
                    assert l.arrival_time >= r.locations[i - 1].departure_time, f"Route of caregiver {r.caregiver_id} is not consistent with the times"
        return self
    
    def check_validity(self, instance : Instance):
        # check that all patients are visited
        patients = {l.patient for r in self.routes for l in r.locations} 
        visited_patients = {p.id for p in instance.patients}
        assert patients == visited_patients, f"Some patients are not visited ({visited_patients - patients})"
        # check that all services required by each patient are provided
        for p in instance.patients:
            for s in p.required_services:
                providing_caregivers = {r.caregiver_id for r in self.routes for l in r.locations if l.patient == p.id and l.service == s.service}
                assert len(providing_caregivers) >= 1, f"Patient {p.id} requires service {s.service} which is not provided"
                assert len(providing_caregivers) == 1, f"Patient {p.id} requires service {s.service} which is provided by more than one caregiver"
        # check that the times are consistent with the distances
        for r in self.routes:
            for i in range(len(r.locations) - 1):
                start_index = next(p.distance_matrix_index for p in instance.patients if p.id == r.locations[i].patient)
                end_index = next(p.distance_matrix_index for p in instance.patients if p.id == r.locations[i + 1].patient)
                assert r.locations[i + 1].arrival_time >= r.locations[i].departure_time + instance.distances[start_index][end_index], f"Route of caregiver {r.caregiver_id} is not consistent with the distances"
            # check that the times are consistent with the service duration
            for l in r.locations:
                p = next(p for p in instance.patients if p.id == l.patient)
                s = next(s for s in p.required_services if s.service == l.service)
                assert l.departure_time - l.arrival_time >= s.actual_duration, f"Route of caregiver {r.caregiver_id} is not consistent with the service durations"
            # check that the times are consistent with the working shift
            c = next(c for c in instance.caregivers if c.id == r.caregiver_id)
            assert r.locations[0].arrival_time >= c.working_shift[0], f"Route of caregiver {r.caregiver_id} is not consistent with his/her working shift"
        # check that for patients requiring sequential and simultaneous services, the services are provided correctly
        for p in instance.patients:            
            if p.synchronization is not None:
                caregivers = {l.service: l for r in self.routes for l in r.locations if l.patient == p.id}
                assert len(caregivers) == 2, f"Patient {p.id} requires double service but they are provided by the same caregiver"
                if p.synchronization.type == 'simultaneous':
                    assert caregivers[p.required_services[0].service].arrival_time == caregivers[p.required_services[0].service].arrival_time, f"Patient {p.id} requires simultaneous service but they are not provided simultaneously {caregivers[p.required_services[0].service].arrival_time} vs {caregivers[p.required_services[1].service].arrival_time}"
                else:
                    assert caregivers[p.required_services[0].service].arrival_time + p.synchronization.distance[0] <= caregivers[p.required_services[1].service].arrival_time, f"Patient {p.id} requires sequential service but the order is not respected (second service starting too early {caregivers[p.required_services[0].service].arrival_time} vs {caregivers[p.required_services[1].service].arrival_time} and min distance {p.synchronization.distance[0]})"
                    assert caregivers[p.required_services[0].service].arrival_time + p.synchronization.distance[1] >= caregivers[p.required_services[1].service].arrival_time, f"Patient {p.id} requires sequential service but the order is not respected (second service starting too late {caregivers[p.required_services[0].service].arrival_time} vs {caregivers[p.required_services[1].service].arrival_time} and max distance {p.synchronization.distance[1]})"
        # check that a patient is served after his/her time window starts
        for p in instance.patients:
            for r in self.routes:
                for l in r.locations:
                    if l.patient == p.id:
                        assert l.arrival_time >= p.time_window[0], f"Patient {p.id} is served before his/her time window starts"
        # check that all caregivers provide services for whih they are qualified
        for r in self.routes:
            for l in r.locations:
                c = next(c for c in instance.caregivers if c.id == r.caregiver_id)
                assert l.service in c.abilities, f"Caregiver {c.id} is providing a service for which he/she is not qualified"        


    def compute_costs(self, instance : Instance) -> float:
        tardiness = []
        distance_traveled = 0        
        idle_time = []
        extra_time = []
        for r in self.routes:
            c = next(c for c in instance.caregivers if c.id == r.caregiver_id)
            for l in r.locations:
                p = next(p for p in instance.patients if p.id == l.patient)
                tardiness.append(max(0, l.arrival_time - p.time_window[1]))
            # search for the depot which is the first location of the caregier
            depot = next(d for d in instance.departing_points if d.id == c.departing_point)
            prev_location = depot
            arrival_at_patient_time = c.working_shift[0]
            for i, l in enumerate(r.locations):
                current_location = next(p for p in instance.patients if p.id == l.patient)
                distance = instance.distances[prev_location.distance_matrix_index][current_location.distance_matrix_index]
                distance_traveled += distance
                if i == 0:
                    # for the first patient, the caregiver can arrive at the patient location at the earliest
                    arrival_at_patient_time = max(arrival_at_patient_time + distance, l.arrival_time)
                else:
                    arrival_at_patient_time = r.locations[i - 1].departure_time + distance
                if arrival_at_patient_time < l.arrival_time:
                    idle_time.append(l.arrival_time - arrival_at_patient_time)
                prev_location = current_location
            distance = instance.distances[current_location.distance_matrix_index][depot.distance_matrix_index]
            distance_traveled += distance
            arrival_at_depot = arrival_at_patient_time + distance
            if arrival_at_depot > c.working_shift[1]:
                extra_time.append(arrival_at_depot - c.working_shift[1])
        services_to_provide = sum(len(p.required_services) for p in instance.patients)
        min_load = math.floor(services_to_provide / len(instance.caregivers))
        max_load = math.ceil(services_to_provide / len(instance.caregivers))
        under_load, over_load = 0, 0
        for r in self.routes:
            if len(r.locations) < min_load:
                under_load += min_load - len(r.locations)
            if len(r.locations) > max_load:
                over_load += len(r.locations) - max_load

        return {
            'total_tardiness': sum(tardiness), 
            'max_tardiness': max(tardiness) if tardiness else 0,
            'traveled_distance': distance_traveled,
            'total_idle_time': sum(idle_time),
            'max_idle_time': max(idle_time),
            'total_extra_time': sum(extra_time),
            'under_load': under_load,
            'over_load': over_load,
            'balance': under_load + over_load
        }