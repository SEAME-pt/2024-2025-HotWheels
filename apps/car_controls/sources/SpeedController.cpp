/*!
 * @file SpeedController.cpp
 * @brief Implementação do controlador de velocidade inteligente
 * @version 1.0
 * @date 2025-08-21
 * @details Implementa controle PID para velocidade suave e constante
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "SpeedController.hpp"
#include <QDebug>
#include <cmath>
#include <numeric>
#include <algorithm>

SpeedController::SpeedController(QObject *parent)
    : QObject(parent),
      m_targetStraightSpeed(1.5f),  // Velocidade mais conservadora e realista
      m_targetTurnSpeed(1.0f),      // Velocidade reduzida para curvas  
      m_targetSpeed(0.0f),
      m_kp(15.0f),                  // Ganho proporcional muito maior para resposta forte
      m_ki(3.0f),                   // Ganho integral maior para eliminar erro permanente
      m_kd(1.0f),                   // Ganho derivativo para controlar oscilações
      m_integralError(0.0f),
      m_previousError(0.0f),
      m_lastUpdateTime(std::chrono::steady_clock::now())
{
    // Inicializa histórico de throttle com zeros
    m_throttleHistory.resize(HISTORY_SIZE, 0);
    
    qDebug() << "[SpeedController] Initialized with HIGH-TORQUE PID gains: Kp=" << m_kp 
             << ", Ki=" << m_ki << ", Kd=" << m_kd;
    qDebug() << "[SpeedController] Target speeds (ground-optimized): Straight=" << m_targetStraightSpeed 
             << " km/h, Turn=" << m_targetTurnSpeed << " km/h";
    qDebug() << "[SpeedController] Friction compensation: Static boost=" << STATIC_FRICTION_BOOST
             << ", Dynamic offset=" << DYNAMIC_FRICTION_OFFSET;
}

SpeedController::~SpeedController() {
    qDebug() << "[SpeedController] Destructor called";
}

int SpeedController::calculateThrottle(float targetSpeed, float currentSpeed, bool isTurning) {
    // === DETECÇÃO DE ESTADO STATIONARY COM HYSTERESIS ===
    // Evidência empírica: transições muito frequentes causam "soluços"
    // Solução: Hysteresis - diferentes limites para entrada e saída do estado
    static bool wasStationary = true;  // Inicia como parado
    bool isStationary;
    
    if (wasStationary) {
        // Para SAIR do estado STATIONARY: velocidade deve ser maior que 0.15 km/h
        isStationary = (std::abs(currentSpeed) < 0.15f);
    } else {
        // Para ENTRAR no estado STATIONARY: velocidade deve ser menor que 0.05 km/h  
        isStationary = (std::abs(currentSpeed) < 0.05f);
    }
    
    wasStationary = isStationary;  // Salva estado para próxima iteração
    
    // === ZONA DE TOLERÂNCIA: se dentro de ±0.2 km/h do target, mantém throttle anterior ===
    static int lastThrottle = 0;
    // Só aplica tolerância se já estiver acima de 80% da targetSpeed
    if (currentSpeed > 0.8f * targetSpeed && std::abs(currentSpeed - targetSpeed) < 0.1f) {
        // Mantém o último throttle para evitar correções desnecessárias
        return lastThrottle;
    }
    // PID atua normalmente para garantir aceleração até o alvo
    int throttle = calculateThrottleWithFrictionCompensation(targetSpeed, currentSpeed, isTurning, isStationary);
    lastThrottle = throttle;
    return throttle;
}

int SpeedController::calculateThrottleWithFrictionCompensation(float targetSpeed, float currentSpeed, 
                                                              bool isTurning, bool isStationary) {
    // Atualiza velocidade alvo baseada no contexto
    if (targetSpeed > 0) {
        m_targetSpeed = targetSpeed;
    } else {
        m_targetSpeed = isTurning ? m_targetTurnSpeed : m_targetStraightSpeed;
    }

    // Calcula tempo decorrido
    auto currentTime = std::chrono::steady_clock::now();
    auto deltaTime = std::chrono::duration<float>(currentTime - m_lastUpdateTime).count();
    m_lastUpdateTime = currentTime;

    // Evita divisão por zero ou valores muito pequenos
    if (deltaTime < 0.001f) deltaTime = 0.1f;

    // Calcula erro de velocidade
    float error = m_targetSpeed - currentSpeed;

    // === Controle PID ===
    
    // Termo Proporcional (aumentado para motores com carga)
    float proportional = m_kp * error;

    // Termo Integral (com limitação anti-windup mais generosa)
    m_integralError += error * deltaTime;
    m_integralError = clamp(m_integralError, -MAX_INTEGRAL, MAX_INTEGRAL);
    float integral = m_ki * m_integralError;

    // Termo Derivativo
    float derivative = m_kd * (error - m_previousError) / deltaTime;
    m_previousError = error;

    // Soma dos termos PID
    float pidOutput = proportional + integral + derivative;

    // === COMPENSAÇÃO DE ATRITO - A PARTE CRÍTICA! ===
    
    int baseThrottle = static_cast<int>(std::round(pidOutput));
    //! int compensatedThrottle = baseThrottle;
    
    // Se queremos movimento para frente (velocidade > 0)
    static int curveBoostCycles = 0;

    //! MELO 
    static int compensatedThrottle = MIN_MOVING_THROTTLE;
    // bool isSharpCurve = (isTurning && std::abs(error) > 0.5f);
    if (m_targetSpeed > 0.1f) {
        if (currentSpeed == 0.0f){
            compensatedThrottle = MIN_MOVING_THROTTLE;
        }
        else if (error < 0){
            compensatedThrottle = compensatedThrottle-2;
            //compensatedThrottle--;
        }
        else if (error > 0.2f){
            compensatedThrottle++;
        }
    }
    // if (m_targetSpeed > 0.1f) {
    //     if (isStationary && error > 0.3f) {
    //         compensatedThrottle = std::max(baseThrottle, STATIC_FRICTION_BOOST);
    //         qDebug() << "[SpeedController] STATIC FRICTION BOOST applied:" << STATIC_FRICTION_BOOST;
    //     } else if (!isStationary && baseThrottle > 0 && baseThrottle < MIN_MOVING_THROTTLE) {
    //         int minThrottle = MIN_MOVING_THROTTLE;
    //         if (isSharpCurve) minThrottle = MIN_MOVING_THROTTLE + 5;
    //         compensatedThrottle = minThrottle;
    //         qDebug() << "[SpeedController] MIN_MOVING_THROTTLE enforced:" << minThrottle;
    //     } else if (!isStationary && baseThrottle > 0) {
    //         compensatedThrottle = baseThrottle + DYNAMIC_FRICTION_OFFSET;
    //     }
    //     // BOOST temporário ao entrar em curva fechada
    //     if (isSharpCurve && curveBoostCycles < 5) {
    //         compensatedThrottle += 10;
    //         curveBoostCycles++;
    //         qDebug() << "[SpeedController] Curve BOOST applied:" << compensatedThrottle;
    //     } else if (!isSharpCurve) {
    //         curveBoostCycles = 0;
    //     }
    //     // Se conseguimos algum movimento, reduz o boost gradualmente
    //     if (currentSpeed > 0.2f && compensatedThrottle == STATIC_FRICTION_BOOST) {
    //         compensatedThrottle = static_cast<int>(baseThrottle + DYNAMIC_FRICTION_OFFSET);
    //     }
    // }
    // PID mais agressivo em curvas fechadas
    // if (isSharpCurve) {
    //     float aggressiveKp = m_kp * 1.3f;
    //     float aggressiveKd = m_kd * 1.5f;
    //     float proportional = aggressiveKp * error;
    //     float derivative = aggressiveKd * (error - m_previousError) / deltaTime;
    //     float pidOutput = proportional + m_ki * m_integralError + derivative;
    //     int aggressiveThrottle = static_cast<int>(std::round(pidOutput));
    //     compensatedThrottle = std::max(compensatedThrottle, aggressiveThrottle);
    // }
    // // Redução em curvas mais suave para manter movimento
    // if (isTurning && compensatedThrottle > 0) {
    //     compensatedThrottle = static_cast<int>(compensatedThrottle * 0.97f);  // Reduz apenas 3%
    // }

    // // Limitação de segurança (aumentada para lidar com peso)
    // compensatedThrottle = clamp(compensatedThrottle, -MAX_THROTTLE, MAX_THROTTLE);

    // // Suavização temporal (mas preserva força inicial)
    // int smoothedThrottle = smoothThrottle(compensatedThrottle);
    
    // // Se estamos aplicando boost inicial, não suaviza demais
    // if (isStationary && error > 0.2f && smoothedThrottle < compensatedThrottle * 0.8f) {
    //     smoothedThrottle = static_cast<int>(compensatedThrottle * 0.9f); // Suaviza menos
    // }

    // // Debug detalhado para diagnóstico
    // if (std::abs(error) > 0.1f || std::abs(smoothedThrottle) > 10) {
        qDebug() << QString("[SpeedController] Target: %1 km/h, Current: %2 km/h, "
                           "Error: %3, PID: P=%4 I=%5 D=%6, Base: %7, Compensated: %8, Final: %9 %10")
                    .arg(m_targetSpeed, 0, 'f', 1)
                    .arg(currentSpeed, 0, 'f', 1)  
                    .arg(error, 0, 'f', 2)
                    .arg(proportional, 0, 'f', 1)
                    .arg(integral, 0, 'f', 1)
                    .arg(derivative, 0, 'f', 1)
                    .arg(baseThrottle)
                    .arg(compensatedThrottle)
                    .arg(isStationary ? "[STATIONARY]" : "");
    // }

    //return smoothedThrottle;
    return compensatedThrottle;
}

void SpeedController::setTargetStraightSpeed(float speed) {
    m_targetStraightSpeed = clamp(speed, 0.3f, 3.0f);  // Limites mais realistas para chão
    qDebug() << "[SpeedController] Straight speed set to:" << m_targetStraightSpeed << "km/h";
}

void SpeedController::setTargetTurnSpeed(float speed) {
    m_targetTurnSpeed = clamp(speed, 0.2f, 2.0f);      // Limites para curvas no chão
    qDebug() << "[SpeedController] Turn speed set to:" << m_targetTurnSpeed << "km/h";
}

void SpeedController::setPIDGains(float kp, float ki, float kd) {
    m_kp = clamp(kp, 0.0f, 100.0f);  // Permite ganhos muito maiores
    m_ki = clamp(ki, 0.0f, 20.0f);   // Integral mais potente
    m_kd = clamp(kd, 0.0f, 10.0f);   // Derivativo mais forte
    
    qDebug() << "[SpeedController] PID gains updated: Kp=" << m_kp 
             << ", Ki=" << m_ki << ", Kd=" << m_kd;
}

void SpeedController::resetPID() {
    m_integralError = 0.0f;
    m_previousError = 0.0f;
    m_lastUpdateTime = std::chrono::steady_clock::now();
    
    // Limpa histórico de throttle
    std::fill(m_throttleHistory.begin(), m_throttleHistory.end(), 0);
    
    qDebug() << "[SpeedController] PID state reset";
}

int SpeedController::smoothThrottle(int newThrottle) {
    // Adiciona novo valor ao histórico
    m_throttleHistory.push_back(newThrottle);
    if (m_throttleHistory.size() > HISTORY_SIZE) {
        m_throttleHistory.pop_front();
    }

    // Para valores altos (boost inicial), suaviza MUITO menos
    bool isHighThrottle = std::abs(newThrottle) >= STATIC_FRICTION_BOOST;
    
    if (isHighThrottle) {
        // Permite boost completo na primeira aplicação para vencer atrito
        if (m_throttleHistory.size() >= 2) {
            int lastThrottle = m_throttleHistory[m_throttleHistory.size() - 2];
            // Se estava parado e agora precisa de boost, aplica imediatamente
            if (std::abs(lastThrottle) < MIN_MOVING_THROTTLE) {
                return newThrottle;  // Sem suavização para sair da inércia
            }
            // Se já estava em movimento, suaviza muito pouco
            int smoothed = static_cast<int>((newThrottle * 0.8f) + (lastThrottle * 0.2f));
            return smoothed;
        }
        return newThrottle;
    }
    
    // Para valores médios (movimento normal), suavização moderada
    bool isMediumThrottle = std::abs(newThrottle) >= MIN_MOVING_THROTTLE;
    if (isMediumThrottle && m_throttleHistory.size() >= 2) {
        int lastThrottle = m_throttleHistory[m_throttleHistory.size() - 2];
        int smoothed = static_cast<int>((newThrottle * 0.6f) + (lastThrottle * 0.4f));
        return smoothed;
    }
    
    // Suavização completa apenas para valores baixos
    float weightedSum = 0.0f;
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < m_throttleHistory.size(); ++i) {
        float weight = static_cast<float>(i + 1);  
        weightedSum += m_throttleHistory[i] * weight;
        totalWeight += weight;
    }

    int smoothed = static_cast<int>(std::round(weightedSum / totalWeight));

    // Limita mudanças bruscas apenas para valores baixos (máximo ±20 por ciclo)
    if (m_throttleHistory.size() >= 2) {
        int lastThrottle = m_throttleHistory[m_throttleHistory.size() - 2];
        int maxChange = 20;  // Permite mudanças mais rápidas
        smoothed = clamp(smoothed, lastThrottle - maxChange, lastThrottle + maxChange);
    }

    return smoothed;
}
