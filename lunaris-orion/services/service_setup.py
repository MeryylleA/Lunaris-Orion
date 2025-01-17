import subprocess
import sys
import logging
from pathlib import Path
import os
import time
import psutil
import signal

logger = logging.getLogger(__name__)

class ServiceSetup:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.processes = {}
    
    def _start_process(self, name: str, cmd: list):
        """Inicia um processo e monitora seu estado."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes[name] = process
            logger.info(f"Serviço {name} iniciado (PID: {process.pid})")
            return process
        except Exception as e:
            logger.error(f"Erro ao iniciar {name}: {str(e)}")
            raise
    
    def start_api(self):
        """Inicia a API REST."""
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "4",
            "--log-level", "info"
        ]
        return self._start_process("api", cmd)
    
    def start_bot(self):
        """Inicia o bot do Discord."""
        cmd = [
            sys.executable,
            str(self.root_dir / "discord_bot" / "bot.py")
        ]
        return self._start_process("bot", cmd)
    
    def start_monitoring(self):
        """Inicia serviços de monitoramento."""
        # Prometheus
        prometheus_cmd = [
            "prometheus",
            "--config.file=monitoring/prometheus/prometheus.yml"
        ]
        self._start_process("prometheus", prometheus_cmd)
        
        # Grafana
        grafana_cmd = [
            "grafana-server",
            "--config=monitoring/grafana/grafana.ini"
        ]
        self._start_process("grafana", grafana_cmd)
    
    def start_services(self):
        """Inicia todos os serviços necessários."""
        try:
            logger.info("Iniciando serviços...")
            
            # Inicia API
            self.start_api()
            time.sleep(2)  # Aguarda API iniciar
            
            # Inicia Bot
            self.start_bot()
            time.sleep(1)
            
            # Inicia Monitoramento
            self.start_monitoring()
            
            # Monitora processos
            while True:
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        logger.error(f"Serviço {name} parou inesperadamente")
                        # Tenta reiniciar
                        if name == "api":
                            self.start_api()
                        elif name == "bot":
                            self.start_bot()
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Encerrando serviços...")
            self.stop_services()
        except Exception as e:
            logger.error(f"Erro nos serviços: {str(e)}")
            self.stop_services()
            raise
    
    def stop_services(self):
        """Para todos os serviços em execução."""
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Parando {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill() 