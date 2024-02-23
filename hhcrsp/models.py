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
    # FIXME: working shift (in the generator) should be already mapped to the suitable type list(map(int, self.working_shift))
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
    _actual_duration: Annotated[int, Field(gt=0, exclude=True, repr=False)] = None

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
    # FIXME: (in the geenrator) compute times outside the function [[int(d.total_seconds() // 60) for d in r] for r in distances]
    distances: Sequence[Sequence[int]]
    departing_points : Sequence[DepartingPoint]
    caregivers : Sequence[Caregiver]
    patients : Sequence[Patient]
    services : Sequence[Service]
    _departing_points: Annotated[dict[str, DepartingPoint], Field(exclude=True, repr=False)] = None
    _caregivers: Annotated[dict[str, Caregiver], Field(exclude=True, repr=False)] = None
    _patients: Annotated[dict[str, Patient], Field(exclude=True, repr=False)] = None
    _services: Annotated[dict[str, Service], Field(exclude=True, repr=False)] = None

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
                s._actual_duration = s.duration or services[s.service].default_duration                
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
        # checking that the time window is compatible with the service time distance in case of sequential services
        for p in self.patients:
            for s in p.required_services:
                if p.synchronization and p.synchronization.type == 'sequential':
                    assert p.time_window[1] - p.time_window[0] >= min(p.synchronization.distance), f"Patient {p.id} has a time window too short for the synchronization services required"
        
        # fill in the dictionaries for quicker access
        self._departing_points = { dp.id: dp for dp in self.departing_points }
        self._caregivers = { c.id: c for c in self.caregivers }
        self._patients = { p.id: p for p in self.patients }        
        self._services = { s.id: s for s in self.services }
       
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
            service_length += [s._actual_duration for s in p.required_services]
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
    
### Solution Model

class PatientVisit(BaseModel):
    start_service_time: Annotated[int | float, Field(ge=0, alias=AliasChoices('start_service_time', 'arrival_time'))]
    end_service_time: Annotated[int | float, Field(ge=0, alias=AliasChoices('end_service_time', 'departure_time'))]
    patient: str
    service: str
    arrival_at_patient: Optional[int | float] = None

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        assert self.start_service_time < self.end_service_time, f"Start service time for service {self.service} at patient {self.patient} is greater than end service time"
        if self.arrival_at_patient is not None:
            assert self.arrival_at_patient <= self.start_service_time, f"Arrival at patient {self.patient} for service {self.service} is greater than start service time"
        return self
    
class DepotDeparture(BaseModel):
    departing_time: Annotated[int | float, Field(ge=0)]
    depot: str    

class DepotArrival(BaseModel):
    arrival_time: Annotated[int | float, Field(ge=0)]
    depot: str   

class CaregiverRoute(BaseModel):
    caregiver_id: str
    locations: list[DepotDeparture | PatientVisit | DepotArrival]
    _visits: Annotated[list[PatientVisit], Field(exclude=True, repr=False)] = []
    _full_route: Annotated[bool, Field(exclude=True, repr=False)] = False

    @model_validator(mode='after')
    def _check_vallidity(self):
        if self.locations:
            self._full_route = False
            if type(self.locations[0]) == DepotDeparture:
                assert type(self.locations[-1]) == DepotArrival, "First location is a depot but last location is not"
                self._full_route = True
            if self._full_route:                
                assert len(self.locations) > 2, f"Caregiver {self.caregiver_id} has a full route with no intermediate location"            
                self._visits = [self.locations[i] for i in range(1, len(self.locations) - 1)]
            else:
                self._visits = [l for l in self.locations]
            prev_time = 0 if not self._full_route else self.locations[0].departing_time
            for i, l in enumerate(self._visits):                
                assert type(l) == PatientVisit, f"Location {l} of caregiver {self.caregiver_id} (at step {i + 1 if self._full_route else i}) is not a patient location"
                assert l.start_service_time >= prev_time, f"Start service time of caregiver {self.caregiver_id} (at step {i + 1 if self._full_route else i}) {l.start_service_time} is not consistent with the previous one ({prev_time})"
                assert l.end_service_time  > l.start_service_time, f"End service time of caregiver {self.caregiver_id} (at step {i + 1 if self._full_route else i}) {l.end_service_time} is not greater than start service time {l.start_service_time}"
                prev_time = l.end_service_time
            if self._full_route:
                assert self.locations[-1].arrival_time >= prev_time, f"Arrival time at depot of caregiver {self.caregiver_id} {self.locations[-1].arrival_time} is not consistent with the previous end service time {prev_time}"

        return self    

class Solution(BaseModel):
    instance: Optional[Annotated[str, Field(min_length=1)]] = None
    routes: list[CaregiverRoute]
    _normalized: Annotated[bool, Field(exclude=True, repr=False)] = False

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        assert len(self.routes) == len(set(r.caregiver_id for r in self.routes)), "Some caregivers are repeated in the solution"
        self._normalized = True
        for r in self.routes:            
            if any(l.arrival_at_patient is not None for l in r._visits):
                self._normalized = False
                break

        return self
    
    def check_validity(self, instance : Instance):
        # check that all patients are visited
        patients = {l.patient for r in self.routes for l in r._visits} 
        visited_patients = {p.id for p in instance.patients}
        assert patients == visited_patients, f"Some patients are not visited ({visited_patients - patients})"
        # check that all services required by each patient are provided
        for p in instance.patients:
            for s in p.required_services:
                providing_caregivers = {r.caregiver_id for r in self.routes for l in r._visits if l.patient == p.id and l.service == s.service}
                assert len(providing_caregivers) >= 1, f"Patient {p.id} requires service {s.service} which is not provided"
                assert len(providing_caregivers) == 1, f"Patient {p.id} requires service {s.service} which is provided by more than one caregiver"
        # normalize routes, by transforming them into full routes
        for r in self.routes:
            c = instance._caregivers[r.caregiver_id]
            if r._full_route:
                continue
            # search for the depot which is the first location of the caregiver
            depot = instance._departing_points[c.departing_point]            
            # search for the patient which is the first location of the caregiver
            first_patient = instance._patients[r._visits[0].patient]
            travel_time = instance.distances[depot.distance_matrix_index][first_patient.distance_matrix_index]
            # compute the latest time the caregiver can depart to arrive at first patient
            depot_departure = r.locations[0].start_service_time - travel_time
            r.locations.insert(0, DepotDeparture(departing_time=depot_departure, depot=depot.id))
            # compute the earliest time the caregiver arrive at the depot after the last patient
            last_patient = instance._patients[r._visits[-1].patient]
            travel_time = instance.distances[last_patient.distance_matrix_index][depot.distance_matrix_index]
            depot_arrival = r._visits[-1].end_service_time + travel_time
            r.locations.append(DepotArrival(arrival_time=depot_arrival, depot=depot.id))
            r._full_route = True

        # check that the times are consistent with the distances and normalize the arrival times
        for r in self.routes:
            for i in range(len(r._visits) - 1):
                start_index = instance._patients[r._visits[i].patient].distance_matrix_index
                end_index = instance._patients[r._visits[i + 1].patient].distance_matrix_index
                travel_time = instance.distances[start_index][end_index]
                assert r._visits[i + 1].start_service_time >= r._visits[i].end_service_time + travel_time, f"Route of caregiver {r.caregiver_id} is not consistent with the distances ({r.locations[i].end_service_time} + {travel_time} vs {r.locations[i + 1].start_service_time})"
                if not r._visits[i + 1].arrival_at_patient:
                    r._visits[i + 1].arrival_at_patient = r._visits[i].end_service_time + travel_time
                assert r._visits[i + 1].start_service_time >= r._visits[i + 1].arrival_at_patient, f"Route of caregiver {r.caregiver_id} is not consistent with the arrival at patient"                
            # check the first location (i.e., depot)
            start_index = instance._departing_points[r.locations[0].depot].distance_matrix_index
            end_index = instance._patients[r.locations[1].patient].distance_matrix_index
            travel_time = instance.distances[start_index][end_index]
            assert r.locations[1].start_service_time >= r.locations[0].departing_time + travel_time, f"Route of caregiver {r.caregiver_id} is not consistent with the distances from the depot ({r.locations[0].departing_time} + {travel_time} vs {r.locations[1].start_service_time})"
            if not r.locations[1].arrival_at_patient:
                r.locations[1].arrival_at_patient = r.locations[0].departing_time + travel_time
            assert r.locations[1].start_service_time >= r.locations[1].arrival_at_patient, f"Route of caregiver {r.caregiver_id} is not consistent with the arrival at patient"
            # check the last location (i.e., depot)
            start_index = instance._patients[r.locations[-2].patient].distance_matrix_index
            end_index = instance._departing_points[r.locations[-1].depot].distance_matrix_index
            travel_time = instance.distances[start_index][end_index]
            assert r.locations[-1].arrival_time >= r.locations[-2].end_service_time + travel_time, f"Route of caregiver {r.caregiver_id} is not consistent with the distances to the depot ({r.locations[-2].end_service_time} + {travel_time} vs {r.locations[-1].arrival_time})"
            # check that the times are consistent with the service duration
            for l in r._visits:
                s = next(s for s in instance._patients[l.patient].required_services if s.service == l.service)
                assert l.end_service_time - l.start_service_time >= s._actual_duration, f"Route of caregiver {r.caregiver_id} at {l} is not consistent with the service durations"
            # check that the times are consistent with the working shift
            c = instance._caregivers[r.caregiver_id]
            assert r.locations[0].departing_time >= c.working_shift[0], f"Route of caregiver {r.caregiver_id} is not consistent with his/her working shift"
        # check that for patients requiring sequential and simultaneous services, the services are provided correctly
        for p in instance.patients:            
            if p.synchronization is not None:
                caregivers = {l.service: l for r in self.routes for l in r._visits if l.patient == p.id}
                assert len(caregivers) == 2, f"Patient {p.id} requires double service but they are provided by the same caregiver"
                if p.synchronization.type == 'simultaneous':
                    assert caregivers[p.required_services[0].service].start_service_time == caregivers[p.required_services[0].service].start_service_time, f"Patient {p.id} requires simultaneous service but they are not provided simultaneously {caregivers[p.required_services[0].service].start_service_time} vs {caregivers[p.required_services[1].service].start_service_time}"
                else:
                    assert caregivers[p.required_services[0].service].start_service_time + p.synchronization.distance[0] <= caregivers[p.required_services[1].service].start_service_time, f"Patient {p.id} requires sequential service but the order is not respected (second service starting too early {caregivers[p.required_services[0].service].start_service_time} vs {caregivers[p.required_services[1].service].start_service_time} and min distance {p.synchronization.distance[0]})"
                    assert caregivers[p.required_services[0].service].start_service_time + p.synchronization.distance[1] >= caregivers[p.required_services[1].service].start_service_time, f"Patient {p.id} requires sequential service but the order is not respected (second service starting too late {caregivers[p.required_services[0].service].start_service_time} vs {caregivers[p.required_services[1].service].start_service_time} and max distance {p.synchronization.distance[1]})"
        # check that a patient is served after his/her time window starts
        for p in instance.patients:
            for r in self.routes:
                for l in r._visits:
                    if l.patient == p.id:
                        assert l.start_service_time >= p.time_window[0], f"Patient {p.id} is served before his/her time window starts"
        # check that all caregivers provide services for whih they are qualified
        for r in self.routes:
            for l in r._visits:
                c = instance._caregivers[r.caregiver_id]
                assert l.service in c.abilities, f"Caregiver {c.id} is providing a service for which he/she is not qualified"        


    def compute_costs(self, instance : Instance) -> float:
        tardiness = []
        distance_traveled = 0        
        waiting_time = []
        extra_time = []
        for r in self.routes:
            c = instance._caregivers[r.caregiver_id]
            # compute tardiness
            for l in r._visits:
                p = instance._patients[l.patient]
                tardiness.append(max(0, l.start_service_time - p.time_window[1]))
            # compute distance traveled
            start_index = instance._departing_points[c.departing_point].distance_matrix_index
            for i, l in enumerate(r._visits):
                end_index = instance._patients[l.patient].distance_matrix_index
                travel_time = instance.distances[start_index][end_index]
                distance_traveled += travel_time
                start_index = end_index
                if l.arrival_at_patient < l.start_service_time:
                    waiting_time.append(l.start_service_time - l.arrival_at_patient)
            end_index = instance._departing_points[c.departing_point].distance_matrix_index
            distance_traveled += instance.distances[start_index][end_index]            

        services_to_provide = sum(len(p.required_services) for p in instance.patients)
        min_load = math.floor(services_to_provide / len(instance.caregivers))
        max_load = math.ceil(services_to_provide / len(instance.caregivers))
        under_load, over_load = 0, 0
        for r in self.routes:
            if len(r.locations) < min_load:
                under_load += min_load - len(r.locations)
            if len(r.locations) > max_load:
                over_load += len(r.locations) - max_load


        for r in self.routes:
            c = instance._caregivers[r.caregiver_id]
            if c.working_shift is not None and r.locations[-1].arrival_time > c.working_shift[1]:
                extra_time.append(r.locations[-1].arrival_time - c.working_shift[1])

        return {
            'total_tardiness': sum(tardiness), 
            'max_tardiness': max(tardiness) if tardiness else 0,
            'traveled_distance': distance_traveled,
            'total_waiting_time': sum(waiting_time),
            'max_waiting_time': max(waiting_time),
            'total_extra_time': sum(extra_time),
            'under_load': under_load,
            'over_load': over_load,
            'balance': under_load + over_load
        }