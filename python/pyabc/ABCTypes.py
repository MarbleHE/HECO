from typing import TypeVar, NewType, List, Generic

# Generic types.
T = TypeVar("T")


class Secret(Generic[T]):
    pass


_created_types = dict()


def NewTypeWithMemory(name, tp):
    _created_types[name] = tp
    return NewType(name, tp)


def get_created_type(name):
    return _created_types.get(name, None)


# Specific types. Those are useful if we want to hint to the translation what value the arguments haves
SecretInt = NewTypeWithMemory("SecretInt", Secret[int])
SecretFloat = NewTypeWithMemory("SecretFloat", Secret[float])
SecretDouble = NewTypeWithMemory("SecretDouble", Secret[float])
SecretBool = NewTypeWithMemory("SecretBool", Secret[bool])
SecretString = NewTypeWithMemory("SecretString", Secret[str])
SecretChar = NewTypeWithMemory("SecretChar", Secret[str])

SecretIntVector = NewTypeWithMemory("SecretIntVector", List[Secret[int]])
SecretFloatVector = NewTypeWithMemory("SecretFloatVector", List[Secret[float]])
SecretDoubleVector = NewTypeWithMemory("SecretDoubleVector", List[Secret[float]])
SecretBoolVector = NewTypeWithMemory("SecretBoolVector", List[Secret[bool]])
SecretStringVector = NewTypeWithMemory("SecretStringVector", List[Secret[str]])
SecretCharVector = NewTypeWithMemory("SecretCharVector", List[Secret[str]])

NonSecretInt = NewTypeWithMemory("NonSecretInt", int)
NonSecretFloat = NewTypeWithMemory("NonSecretFloat", float)
NonSecretDouble = NewTypeWithMemory("NonSecretDouble", float)
NonSecretBool = NewTypeWithMemory("NonSecretBool", bool)
NonSecretString = NewTypeWithMemory("NonSecretString", str)
NonSecretChar = NewTypeWithMemory("NonSecretChar", str)

NonSecretIntVector = NewTypeWithMemory("NonSecretIntVector", List[int])
NonSecretFloatVector = NewTypeWithMemory("NonSecretFloatVector", List[float])
NonSecretDoubleVector = NewTypeWithMemory("NonSecretDoubleVector", List[float])
NonSecretBoolVector = NewTypeWithMemory("NonSecretBoolVector", List[bool])
NonSecretStringVector = NewTypeWithMemory("NonSecretStringVector", List[str])
NonSecretCharVector = NewTypeWithMemory("NonSecretCharVector", List[str])


def is_secret(val_type):
    """
    Return true when a type is secret. All unknown values default to being secret.

    :param val_type: either a Secret* or NonSecret* type as type or String representation of the type.
                 Any other types default to being secret too.
    :return: false if it is a NonSecret* type (as type or string), true otherwise
    """

    non_secret_types = [NonSecret, NonSecretInt, NonSecretFloat, NonSecretDouble, NonSecretBool, NonSecretString,
                        NonSecretChar, NonSecretIntVector,
                        NonSecretFloatVector, NonSecretDoubleVector, NonSecretBoolVector, NonSecretStringVector,
                        NonSecretCharVector]

    # Case 1: type is given as string, check if it corresponds to the string version of a non_secret type
    if isinstance(val_type, str):
        return val_type not in map(lambda x: x.__name__, non_secret_types)
    # Case 2: type is given as TypeVar or NewType, compare it directly to our non-secret types
    elif isinstance(val_type, type(TypeVar)) or isinstance(val_type, type(NewType)):
        return val_type not in non_secret_types
    # Case 3: any other type just defaults to being secret
    else:
        return True
