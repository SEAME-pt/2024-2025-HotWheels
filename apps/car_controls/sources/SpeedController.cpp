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
    
    // Atualiza velocidade alvo baseada no contexto
    if (targetSpeed > 0) m_targetSpeed = targetSpeed;
    else m_targetSpeed = isTurning ? m_targetTurnSpeed : m_targetStraightSpeed;

    // Calcula erro de velocidade
    float error = m_targetSpeed - currentSpeed;
    int turnThrottle = MIN_MOVING_THROTTLE - 5;

    //! MELO 
    static int compensatedThrottle = MIN_MOVING_THROTTLE;
    if (m_targetSpeed > 0.1f) {
        if (currentSpeed == 0.0f){
            compensatedThrottle = MIN_MOVING_THROTTLE;
        }
        else if (error < 0){
            //compensatedThrottle = compensatedThrottle-2;
            compensatedThrottle = (compensatedThrottle-2 < MIN_MOVING_THROTTLE-3) ? turnThrottle : compensatedThrottle-2;
        }
        else if (error > 0.2f){
            compensatedThrottle++;
        }
    }

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
        
/* 
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
 */