"""Recovered cash-flow facts; all yield/buffer assumptions remain explicit.

Source is a Library snapshot of a working Drive document, not an independently
verified immutable official publication. No eligibility or strategy approval.
"""
from decimal import localcontext
from .accounting import Contribution,Obligation,dec

SOURCE_LIBRARY_ID='libfile_a70ab99c9d808191a7b55c44b20742b1'
SOURCE_DOCUMENT_ID='1kjLz0M7yy8jxIEbeUnsFML1dLQ2_elf8U9dqq0Gka-U'
SOURCE_REVISION='ANLCKQkBFdGCXbn9XTFIrvEXlfgfGfMr5dWDRd4KxLdpJVRSpWEokxjysTEefx0OXjuZ2IqP5JEHs2xZfm2CeMPsr9QzUBMyHilfPsKRD5E'

def contributions_for_year(year):
    if not isinstance(year,int) or isinstance(year,bool):raise ValueError('integer year required')
    amounts={2027:300000,2028:150000}
    return (Contribution(f'laura-contribution-{year}',amounts[year]),) if year in amounts else ()

def operating_schedule():
    return tuple(Obligation(f'operating-{year}',year,50000) for year in range(2033,2043))

def required_operating_reserve(reserve_returns):
    """Beginning-2033 reserve; nine explicit interval yields, first payment now."""
    rates=tuple(dec(r) for r in reserve_returns)
    if len(rates)!=9 or min(rates)<=-1:raise ValueError('nine interval returns > -1 required')
    with localcontext() as ctx:
        ctx.prec=40
        reserve=dec(50000)
        for rate in reversed(rates):reserve=dec(50000)+reserve/(1+rate)
        return reserve

def facility_capacity(portfolio,operating_reserve,flexibility_buffer):
    """Mechanical ceiling after reserving operating payments and explicit buffer."""
    values=tuple(dec(x) for x in (portfolio,operating_reserve,flexibility_buffer))
    if min(values)<0:raise ValueError('negative budget')
    wealth,reserve,buffer=values
    return {'mechanical_ceiling':max(dec(0),wealth-reserve-buffer),
            'reserve_and_buffer_gap':max(dec(0),reserve+buffer-wealth),
            'status':'assumption-conditioned arithmetic, not recommended facility commitment'}
