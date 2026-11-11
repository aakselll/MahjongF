"""
Parameter server : contient l'etat partage entre workers et trainer.

[FIX Windows] Le Manager et les proxies sont crees DANS le process
principal. Les workers recoivent UNIQUEMENT les proxies (picklables),
pas l'objet ParameterServer lui-meme.
"""
import multiprocessing as mp
import torch


class ParameterServer:
    """Conteneur des proxies partages. Cree dans le main, jamais envoye
    aux workers directement."""

    def __init__(self):
        # Manager cree dans le process principal
        self.manager = mp.Manager()
        self.state = self.manager.dict()
        self.lock = self.manager.Lock()
        # Init
        self.state['step'] = 0
        self.state['alpha'] = 0.01
        self.state['gamma_t'] = 1.0
        self.state['policy_state'] = None
        self.state['critic_state'] = None

    def publish(self, policy_state, critic_state, step, alpha, gamma_t):
        with self.lock:
            self.state['policy_state'] = {
                k: v.cpu() for k, v in policy_state.items()
            }
            self.state['critic_state'] = {
                k: v.cpu() for k, v in critic_state.items()
            }
            self.state['step'] = step
            self.state['alpha'] = alpha
            self.state['gamma_t'] = gamma_t

    def fetch(self):
        with self.lock:
            return (
                self.state['policy_state'],
                self.state['critic_state'],
                self.state['step'],
                self.state['alpha'],
                self.state['gamma_t'],
            )


# ---------------------------------------------------------------------- #
# Helper functions picklables : les workers recoivent ces fonctions +
# les proxies (dict, lock), pas l'objet ParameterServer.
# ---------------------------------------------------------------------- #
def fetch_from_state(state, lock):
    """Version picklable de fetch() : prend le dict et le lock en argument.

    [FIX 2026-10-14] Les workers demarrent avant la premiere publication
    et (selon le chemin de creation du state) potentiellement avant son
    initialisation. Retourner des valeurs par defaut au lieu de crasher :
    le worker gere deja policy_state is None (tourne sur la policy locale
    warm-startee jusqu'a la premiere publication).
    """
    with lock:
        policy_state = state.get('policy_state')
        critic_state = state.get('critic_state')
        step = state.get('step', 0)
        alpha = state.get('alpha', 0.01)
        gamma_t = state.get('gamma_t', 1.0)
        return (policy_state, critic_state, step, alpha, gamma_t)

def publish_to_state(state, lock, policy_state, critic_state,
                      step, alpha, gamma_t):
    with lock:
        state['policy_state'] = {k: v.cpu() for k, v in policy_state.items()}
        state['critic_state'] = {k: v.cpu() for k, v in critic_state.items()}
        state['step'] = step
        state['alpha'] = alpha
        state['gamma_t'] = gamma_t