from dishka import Provider, Scope, provide

from application.hausman.interactor import HausmanInteractor
from domain.ports.hausman_calculator import HausmanCalculatorPort
from infrastructure.statistics.hausman import HausmanTest


class HausmanProvider(Provider):
    @provide(scope=Scope.APP)
    def hausman_calculator(self) -> HausmanCalculatorPort:
        return HausmanTest(alpha=0.05)

    @provide(scope=Scope.REQUEST)
    def hausman_interactor(
        self,
        calculator: HausmanCalculatorPort,
    ) -> HausmanInteractor:
        return HausmanInteractor(calculator)
