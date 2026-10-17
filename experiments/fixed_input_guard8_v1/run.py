"""Authorized fresh context/pin run, without wall/GPU-time/VRAM admission gates."""
import os
from . import launch
from .configuration import VERSION
from .complete import remove_resource_gates


def main():
    c, root = launch.helpers()
    config = c.read(root/'private/config.json')
    assert config['version'] == VERSION
    assert config['authorization']['execution'] and config['authorization']['resource_gate_waiver']
    assert c.read(root/'public/RESOURCE_GATE_WAIVER.json')['authorized'] is True
    remove_resource_gates()
    if config.get('source6_diagnostic'):
        from .diagnostic import install
        install()
    (launch.child if os.environ['ENTRY_MODE'] == 'child' else launch.controller)()


if __name__ == '__main__':main()
