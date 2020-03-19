from cqt.strats.strategy import Strategy
from cqt.analyze.signal_long_short_crossing import signal_long_short_crossing as slsc
from cqt.analyze.signal_long_short_crossing import signal_average_envelope as sae
import copy

class StrategySimpleMA(Strategy):
    def apply_event_logic(self, time, ledger):
        coin = 'btc'
        if self.env.has_section(coin):
            section_coin = self.env.get_section(coin)
            ind_coin = slsc(self.env, coin, time, self.rules)
            price_coin = section_coin.get_price_close(time)

            if ind_coin == -1:
                ledger.sell_unit(coin, price_coin)
            elif ind_coin == 1:
                ledger.buy(coin, price_coin)
            else:
                pass

        return ledger


class StrategyInverseMA(Strategy):
    def __init__(self, mdl, ini_prtf, rules):
            self.asset_model = mdl
            self.initial_portfolio = ini_prtf
            self.rules = rules

            self.env = mdl
            self.initial = ini_prtf
            self.prices = copy.deepcopy(self.asset_model.get_section('btc').data)
            self.signal=self.prices['price_close'].values*0
            
    def apply_event_logic(self, time, ledger):
            coin = 'btc'
            if self.env.has_section(coin):
                section_coin = self.env.get_section(coin)
                ind_coin = slsc(self.env, coin, time, self.rules)
                price_coin = section_coin.get_price_close(time)
                time_step = self.prices.index.get_loc(time)
                if ind_coin == 1:
                    ledger.sell_unit(coin, price_coin)
                    self.signal[time_step]=1 
                elif ind_coin == -1:
                    ledger.buy(coin, price_coin)
                    self.signal[time_step]=-1 
                else:
                    self.signal[time_step]=0
                    pass

            return ledger


class StrategyBlendMA(Strategy):
    def apply_event_logic(self, time, ledger):
        coin = 'btc'

        if self.env.has_section(coin):
            rules_short = self.rules.copy()
            rules_short['window_size'] = [rules_short['window_size'][0], rules_short['window_size'][1]]
            rules_long = self.rules.copy()
            rules_long['window_size'] = [rules_long['window_size'][2], rules_long['window_size'][3]]

            ind_coin_long = slsc(self.env, coin, time, rules_long)
            ind_coin_short = slsc(self.env, coin, time, rules_short)
            strats_long = StrategySimpleMA(self.env, ledger, rules_long)
            strats_short = StrategySimpleMA(self.env, ledger, rules_short)

            if ind_coin_long == 1:
                ledger = strats_long.apply_event_logic(time, ledger)
            elif ind_coin_short == -1:
                ledger = strats_short.apply_event_logic(time, ledger)
            else:
                pass

        return ledger
