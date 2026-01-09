---
name: Preferencias_Globais_Desenvolvimento
description: Preferências pessoais de programação e diretrizes para todos os projetos
applyTo: "**"
---

# Preferências Globais de Desenvolvimento

## 1. Idioma e Comunicação
- Sempre responda em português brasileiro (pt-BR). 
- No início da iteração, sempre destaque qual o LLM que está sendo executado (ex: GPT-4, Claude 2, etc)
- Forneça explicações claras e didáticas adequadas para profissionais da área de negócios aprendendo programação como hobbie
- Todos os elementos de código (funções, classes, variáveis, arquivos, docstrings) devem ser nomeados em inglês (EN-US).
- Excepcionalmente, adicione comentários alinhados à direita em português para funções mais complexas

## 2. Padrões de Código Python
- Use type hints de forma ampla, em funções, classes e outras situações onde são aplicáveis, sempre seguindo as melhores práticas das PEPs
- Adicione docstrings descritivas em inglês (não repita type hints já descritos nas assinaturas)
- Prefira o uso de logger em vez de print()
- Utilize ferramentas de linting (Flake8 ou Pylint) para garantir conformidade com padrões de estilo

## 3. Arquitetura e Design
- Mantenha o código simples, de fácil entendimento e manutenção
- Mantenha a estrutura funcional do código organizada
- O código não deve parecer algo criado aleatoriamente por uma IA
- Garanta que todas as implementações sejam ponderadas e intencionais
- Preze pela utilização das principais estruturas de design patterns quando aplicável, explicando qual foi utilizado e o motivo da escolha

## 4. Dependências e Bibliotecas
- Sempre que possível, utilize bibliotecas padrão do Python para evitar dependências desnecessárias
- Dê preferência ao Pydantic para modelos de dados e validação de argumentos, entrada e saída quando possível
- Eu tenho como padrão o uso do UV como ferramenta de gestão de dependências.
- As bibliotecas necessárias devem estar instaladas no ambiente virtual criado com o UV. Sempre ative o ambiente antes de executar comandos e testes!

## 5. Performance e Concorrência
- Sempre avalie a necessidade de utilizar código assíncrono, threads ou processos para melhor performance

## 6. Testes
- Use Pytest como ferramenta padrão para testes
- Trabalhe com mocks quando necessário
- Utilize fixtures para manter os códigos de teste mais concisos e organizados

## 7. Segurança e Compatibilidade
- Siga as melhores práticas de segurança ao lidar com dados sensíveis (senhas, informações pessoais)
- Mantenha a compatibilidade com versões recentes do Python, evitando funcionalidades obsoletas ou depreciadas
- Escreva código compatível com múltiplas plataformas (Windows, macOS, Linux)

## 8. Documentação
- Arquivos README devem ser escritos em inglês, claros e objetivos
- Use seções bem definidas, gráficos e recursos de markdown para melhorar a apresentação visual
- Não incluir informações de uso interno (como preferências pessoais) no README
- Avalie se a versão mínima do Python poderia ser anterior à usada no desenvolvimento (considere compatibilidade de códigos e requisitos), caso possível, avalie os arquivos de pyproject.toml

## 9. Abordagem de Trabalho
- Quando eu fizer perguntas, avalie o que eu pedi e eventualmente discuta antes de assumir que estou certo e mudar o código
- Garanta uma abordagem colaborativa e reflexiva nas implementações
- Dê preferência por códigos de fácil manutenção e legibilidade
- No modo agente, primeiro apresente as sugestões e pontos a serem implementados, e só depois implemente as alterações nos códigos

## 10. Code Reviews
- Sempre antes de um commit faça um code review. 
- Atente-se as orientações de linguagem ao realizar o code review, que estão descritas no item idioma e comunicação
