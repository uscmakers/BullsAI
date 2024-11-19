#define PIN_CLK 3
#define PIN_DT 4
#define PIN_BUTTON 5
#define PIN_REED 6
#define DEBOUNCE_DELAY 50

volatile int encoderPos = 0;
unsigned long lastInterruptTime = 0;  // for debouncing

volatile bool lastCLK = LOW;
volatile bool lastDT = LOW;

unsigned long lastStepTime = 0;  // last encoder step
float stepsPerSecond = 0.0;

void setup() {
  // initialize serial communications
  Serial.begin(9600);

  // initialize input pins with pull-up resistors
  pinMode(PIN_CLK, INPUT_PULLUP);
  pinMode(PIN_DT, INPUT_PULLUP);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_REED, INPUT_PULLUP);

  // interrupts for CLK and DT pins
  attachInterrupt(digitalPinToInterrupt(PIN_CLK), handleEncoderCLK, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_DT), handleEncoderDT, CHANGE);
}

void loop() {
  // temporarily disable interrupts to read/write encoderPos
  noInterrupts();
  int pos = encoderPos;
  encoderPos = 0;
  interrupts();

  // if there was at least one encoder step
  if (pos != 0) {
    unsigned long currentTime = micros();
    unsigned long delta = currentTime - lastStepTime;

    if (lastStepTime != 0 && delta > 0) {
      stepsPerSecond = 1000000.0 / delta;

      Serial.print("steps/sec: ");
      Serial.println(stepsPerSecond);
    }

    lastStepTime = currentTime;
  }

}

// Interrupt handler for the CLK pin
void handleEncoderCLK() {
  unsigned long currentTime = millis();

  // debounce interrupts
  if (currentTime - lastInterruptTime > DEBOUNCE_DELAY) {
    bool currentCLK = digitalRead(PIN_CLK);
    bool currentDT = digitalRead(PIN_DT);

    // determine rotation
    if (lastCLK != currentCLK) {
      if (currentCLK == currentDT) {
        encoderPos++;
      }
      else {
        encoderPos--;
      }
    }

    lastCLK = currentCLK;
    lastDT = currentDT;
    lastInterruptTime = currentTime;
  }
}

void handleEncoderDT() {
}
