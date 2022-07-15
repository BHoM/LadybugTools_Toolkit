# from concurrent.futures import ThreadPoolExecutor
# from typing import List

# from ..external_comfort_result import ExternalComfortResult
# from .typology import Typology
# from ..external_comfort_result import TypologyResult


# def calculate_typology_results(
#     typologies: List[Typology], external_comfort_result: ExternalComfortResult
# ) -> List[TypologyResult]:
#     """Create a set of results for a set of typologies.

#     Args:
#         external_comfort_result (ExternalComfortResult): An ExternalComfortResult object containing the results of an external comfort simulation.
#         typologies (List[Typology]): A list of Typology objects to be evaluated.

#     Returns:
#         List[TypologyResult]: A list of typology result objects.
#     """

#     if not all(isinstance(x, Typology) for x in typologies):
#         raise ValueError("Not all elements in list given are Typology objects.")

#     results = []
#     with ThreadPoolExecutor() as executor:
#         for typology in typologies:
#             results.append(
#                 executor.submit(TypologyResult, typology, external_comfort_result)
#             )

#     typology_results = []
#     for result in results:
#         typology_results.append(result.result())

#     return typology_results
