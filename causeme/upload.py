from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


if __name__ == '__main__':
    with open('causeme.keys') as f:
        lines = f.readlines()
        username = lines[0].strip()
        password = lines[1].strip()



    # Create a new instance of the Chrome driver
    # driver = webdriver.Chrome('../../chromedriver.exe')
    #headless
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome('../../chromedriver.exe', options=options)

    # Navigate to the login page
    driver.get('https://causeme.uv.es/login/')

    # Find the email and password fields and enter your credentials
    email_field = driver.find_element(by=By.ID, value='email')
    email_field.send_keys(username)

    password_field = driver.find_element(by=By.ID, value='password')
    password_field.send_keys(password)

    # Find and click the login button
    login_button = driver.find_element(by=By.NAME, value='login')
    login_button.click()

    driver.get('https://causeme.uv.es/upload_results/')



    # Wait for the page to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'predictions'))
    )

    # Select the file
    file_input = driver.find_element(By.ID,'predictions')
    file_input.send_keys('C:/Users/Administrateur/git/D2CPY/generation/results/varmodel-python_maxlags=1_linear-VAR_N-3_T-150.json.bz2')

    # Set the file path using JavaScript

    # Click the upload button
    upload_button = driver.find_element(By.ID,'btnSubmit')
    upload_button.click()

    sleep(5)


    html_content = driver.page_source

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table
    table = soup.find('table', {'class': 'table-condensed'})

    # Get headers
    headers = [header.text for header in table.find_all('th')]

    # Get table rows
    rows = table.find('tbody').find_all('tr')
    data = []
    for row in rows:
        cols = row.find_all('td')
        cols = [element.text.strip() for element in cols]
        data.append(cols)

    # At this point, you should be logged in. You can add additional code here to perform other actions.
    # ...

    # Don't forget to close the browser when you're done!
    driver.quit()

    # Print the result
    import pandas as pd

    # Convert to DataFrame for easier viewing and manipulation
    df = pd.DataFrame(data, columns=headers)
    df.to_csv('results.csv')
    print(df)
