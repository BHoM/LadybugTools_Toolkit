from __future__ import annotations

import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from external_comfort.openfield import OpenfieldResult
from external_comfort.typology import Typology, TypologyResult

def create_typologies(openfield_result: OpenfieldResult, typologies: List[Typology]) -> List[TypologyResult]:
    """Create a set of results for a set of typologies.

    Args:
        openfield_result (OpenfieldResult): An OpenfieldResult object containing the results of an outdoor radiant temperature simulation.
        typologies (List[Typology]): A list of Typology objects to be evaluated.
        

    Returns:
        List[TypologyResult]: A list of typology result objects.
    """

    results = []
    with ThreadPoolExecutor() as executor:
        for typology in typologies:
            results.append(executor.submit(TypologyResult, typology, openfield_result))
    
    t = []
    for r in as_completed(results):
        t.append(r.result())
    
    return t            
