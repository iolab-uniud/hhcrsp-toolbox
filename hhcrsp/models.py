from typing import Optional, Literal, Annotated, Self, Any
from pydantic import BaseModel, Field, model_validator, computed_field, AliasChoices
from collections.abc import Sequence
import hashlib
import click
import numpy as np

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
            'simultaneous': sum(len(p.required_services) == 2 and p.synchronization.type == 'simultaneous' for p in self.patients), 
            'sequential': sum(len(p.required_services) == 2 and p.synchronization.type == 'sequential' for p in self.patients) 
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
    