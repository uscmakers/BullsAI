#define PIN_CLK 3
#define PIN_DT 4
#define PIN_BUTTON 5
#define PIN_REED 2
#define DEBOUNCE_DELAY 20

volatile int encoderPos = 0;
unsigned long lastInterruptTime = 0;  // for debouncing
volatile int counter = 0;

volatile bool lastCLK = LOW;
volatile bool lastDT = LOW;

volatile unsigned long lastReedTriggerTime = 0;
volatile int lastReedCounter = 0;
volatile float reedStepsPerSecond = 0.0;
volatile bool reedSPSUpdated = false;
volatile bool reedClosed = false;

void setup() {
  // initialize serial communications
  Serial.begin(115200);

  // initialize input pins with pull-up resistors
  pinMode(PIN_CLK, INPUT_PULLUP);
  pinMode(PIN_DT, INPUT_PULLUP);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_REED, INPUT_PULLUP);

  // interrupts for CLK and DT pins
  attachInterrupt(digitalPinToInterrupt(PIN_CLK), handleEncoderCLK, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_DT), handleEncoderDT, CHANGE);
  
  attachInterrupt(digitalPinToInterrupt(PIN_REED), handleReedSwitch, FALLING);
}

void loop() {
  // temporarily disable interrupts to read/write encoderPos
  noInterrupts();
  int pos = encoderPos;
  encoderPos = 0;
  interrupts();

  // if there was at least one encoder step
  // if (pos != 0) {
  //   Serial.println(counter);
  // }

  if (reedSPSUpdated) {
    noInterrupts();
    float sps = reedStepsPerSecond;
    reedSPSUpdated = false;
    interrupts();

    Serial.print("Reed SPS: ");
    Serial.println(sps);
  }
}

// interrupt handler for the CLK pin
void handleEncoderCLK() {
  unsigned long currentTime = micros();

  // debounce interrupts
  if (currentTime - lastInterruptTime > (DEBOUNCE_DELAY * 1000)) {
    bool currentCLK = digitalRead(PIN_CLK);
    bool currentDT = digitalRead(PIN_DT);

    // determine rotation
    if (lastCLK != currentCLK) {
      counter++;
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

// interrupt handler for the reed pin
void handleReedSwitch() {
  unsigned long currentTime = micros();

  if (currentTime - lastReedTriggerTime > (DEBOUNCE_DELAY * 1000)) {
    bool reed = digitalRead(PIN_REED);

    if (reed == LOW && !reedClosed) {
      unsigned long delta = currentTime - lastReedTriggerTime;

      if (delta > 0) {
        reedStepsPerSecond = ((counter - lastReedCounter) * 1000000) / delta;
        reedSPSUpdated = true;
      }

      lastReedTriggerTime = currentTime;
      lastReedCounter = counter;

      reedClosed = true;
    }
    else if (reed == HIGH && reedClosed) {
      reedClosed = false;
    }
  }
}
