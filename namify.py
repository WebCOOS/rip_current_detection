
import hashlib
import filetype
from typing import Tuple


def namify_for_content( b: bytes ) -> Tuple[str, str]:

    assert b

    kind = filetype.guess( b )

    assert kind is not None

    hexdigest = hashlib.sha512( b ).hexdigest()

    return ( f"{hexdigest}.{str(kind.extension)}", str( kind.extension ), )
