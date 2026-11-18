"""Blind expert validation of the CV ergonomic assessment pipeline.

``pool``                 candidate frames, risk banding, temporal stability
``allocate``             the two-stratum design, solved by min-cost flow
``sample_frames``        select the frames and build the three-panel composites
``scoring``              observations -> REBA/RULA; the one place scores are composed
``build_expert_package`` assemble a self-contained package per rater
``build_workbook``       the Excel scoring workbook, and reading a returned one
``analyze_ratings``      join returned ratings to ground truth and score agreement

Raters may answer in the browser page or in the workbook.  Both ask the same
questions, and both are read back through ``scoring``, so the choice of tool cannot
change the numbers.
"""
