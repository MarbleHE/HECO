
from typing import TypeVar, NewType

# Generic types. Those rely on Python not enforcing types.
Secret      = TypeVar("Secret")
NonSecret   = TypeVar("NonSecret")

# Specific types. Those are useful if we want to hint to the translation what value the arguments haves
SecretInt       = NewType("SecretInt", int)
SecretFloat     = NewType("SecretFloat", float)
SecretDouble    = NewType("SecretDouble", float)
SecretBool      = NewType("SecretBool", bool)
SecretString    = NewType("SecretString", str)
SecretChar      = NewType("SecretChar", str)

SecretIntVector     = NewType("SecretIntVector", int)
SecretFloatVector   = NewType("SecretFloatVector", float)
SecretDoubleVector  = NewType("SecretDoubleVector", float)
SecretBoolVector    = NewType("SecretBoolVector", bool)
SecretStringVector  = NewType("SecretStringVector", str)
SecretCharVector    = NewType("SecretCharVector", str)


NonSecretInt    = NewType("NonSecretInt", int)
NonSecretFloat  = NewType("NonSecretFloat", float)
NonSecretDouble = NewType("NonSecretDouble", float)
NonSecretBool   = NewType("NonSecretBool", bool)
NonSecretString = NewType("NonSecretString", str)
NonSecretChar   = NewType("NonSecretChar", str)

NonSecretIntVector      = NewType("NonSecretIntVector", int)
NonSecretFloatVector    = NewType("NonSecretFloatVector", float)
NonSecretDoubleVector   = NewType("NonSecretDoubleVector", float)
NonSecretBoolVector     = NewType("NonSecretBoolVector", bool)
NonSecretStringVector   = NewType("NonSecretStringVector", str)
NonSecretCharVector     = NewType("NonSecretCharVector", str)

def is_secret(type):
    """
    Return true when a type is secret. All unknown values default to being secret.

    :param type: either a Secret* or NonSecret* type. Any other types default to being secret too.
    :return: false if it is a NonSecret* type, true otherwise
    """

    return type not in [NonSecret, NonSecretInt, NonSecretFloat, NonSecretDouble, NonSecretBool, NonSecretString, NonSecretChar, NonSecretIntVector,
        NonSecretFloatVector, NonSecretDoubleVector, NonSecretBoolVector, NonSecretStringVector, NonSecretCharVector]