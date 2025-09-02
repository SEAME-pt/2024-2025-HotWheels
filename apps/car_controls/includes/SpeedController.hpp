/*!
 * @file SpeedController.hpp
 * @brief Controlador de velocidade inteligente com PID e controle adaptativo
 * @version 1.0
 * @date 2025-08-21
 * @details Este controlador implementa controle PID para manter velocidade constante,
 * reduzindo ruído do PWM e melhorando a suavidade da condução.
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef SPEED_CONTROLLER_HPP
#define SPEED_CONTROLLER_HPP

#include <QObject>
#include <chrono>
#include <deque>

/*!
 * @brief Controlador inteligente de velocidade com PID
 * @details Esta classe implementa controle PID para manter velocidade constante,
 * com filtragem para reduzir ruído do PWM e algoritmos adaptativos para 
 * diferentes condições de direção.
 */
class SpeedController : public QObject {
    Q_OBJECT

public:
    /*!
     * @brief Construtor do SpeedController
     * @param parent QObject pai
     */
    explicit SpeedController(QObject *parent = nullptr);
    
    /*!
     * @brief Destrutor
     */
    ~SpeedController();

    /*!
     * @brief Calcula o comando de throttle baseado na velocidade desejada e atual
     * @param targetSpeed Velocidade desejada (km/h)
     * @param currentSpeed Velocidade atual medida (km/h)
     * @param isTurning Se o carro está fazendo uma curva
     * @return Valor de throttle otimizado com compensação de atrito (-100 a 100)
     */
    int calculateThrottle(float targetSpeed, float currentSpeed, bool isTurning = false);

    /*!
     * @brief Define a velocidade alvo para retas
     * @param speed Velocidade em km/h
     */
    void setTargetStraightSpeed(float speed);

    /*!
     * @brief Define a velocidade alvo para curvas
     * @param speed Velocidade em km/h
     */
    void setTargetTurnSpeed(float speed);

    /*!
     * @brief Define os parâmetros PID
     * @param kp Ganho proporcional
     * @param ki Ganho integral
     * @param kd Ganho derivativo
     */
    void setPIDGains(float kp, float ki, float kd);

    /*!
     * @brief Reseta o controlador PID (limpa integral e deriva)
     */
    void resetPID();

    /*!
     * @brief Obtém a velocidade alvo atual
     * @return Velocidade alvo em km/h
     */
    float getCurrentTargetSpeed() const { return m_targetSpeed; }

private:
    // === Parâmetros de Velocidade ===
    float m_targetStraightSpeed;    ///< Velocidade alvo para retas (km/h)
    float m_targetTurnSpeed;        ///< Velocidade alvo para curvas (km/h)
    float m_targetSpeed;            ///< Velocidade alvo atual (km/h)

    // === Parâmetros PID ===
    float m_kp;                     ///< Ganho proporcional
    float m_ki;                     ///< Ganho integral
    float m_kd;                     ///< Ganho derivativo

    // === Estado do PID ===
    float m_integralError;          ///< Erro acumulado para termo integral
    float m_previousError;          ///< Erro anterior para termo derivativo
    std::chrono::steady_clock::time_point m_lastUpdateTime; ///< Último tempo de atualização

    // === Filtragem e Suavização ===
    std::deque<int> m_throttleHistory;  ///< Histórico de throttle para suavização
    static const size_t HISTORY_SIZE = 5;  ///< Tamanho do buffer de suavização
    
    // === Parâmetros de Segurança e Força ===
    static const int MAX_THROTTLE = 80;         ///< Throttle máximo para segurança
    static const int MIN_MOVING_THROTTLE = 22;  ///< Throttle mínimo para movimento real no chão
    static const int STATIC_FRICTION_BOOST = 8; ///< Boost extra para vencer atrito estático
    static const int DYNAMIC_FRICTION_OFFSET = 15; ///< Offset constante para atrito dinâmico
    constexpr static const float MAX_INTEGRAL = 100.0f; ///< Limite do termo integral (aumentado)

    // === Métodos Privados ===
    
    /*!
     * @brief Suaviza o valor de throttle usando média móvel
     * @param newThrottle Novo valor de throttle
     * @return Valor suavizado
     */
    int smoothThrottle(int newThrottle);

    /*!
     * @brief Limita o valor dentro de um intervalo
     * @param value Valor a ser limitado
     * @param min Valor mínimo
     * @param max Valor máximo
     * @return Valor limitado
     */
    template<typename T>
    T clamp(T value, T min, T max) const {
        return (value < min) ? min : ((value > max) ? max : value);
    }
};

#endif // SPEED_CONTROLLER_HPP
